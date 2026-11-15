# SNAPRAG: Cost-Efficient Knowledge Extraction Attack on RAG Systems via Adversarial Perturbations

**Authors**: Melih Coşğun, Mehmet Emre Gürsoy, Alptekin Küpçü, Sinem Sav

**Published**: 2026-11-09 11:30:44

**PDF URL**: [https://arxiv.org/pdf/2611.12241v1](https://arxiv.org/pdf/2611.12241v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems improve Large Language Model (LLM) responses by incorporating information from external knowledge bases. However, recent studies have shown that attackers can extract substantial portions of private knowledge bases from RAG systems through repeated querying. Yet, state-of-the-art extraction attacks either rely heavily on external LLMs to generate adversarial queries or require a large number of queries. In this paper, we propose SNAPRAG, a novel and cost-efficient attack for extracting RAG knowledge bases through adversarial perturbations. Our approach algorithmically generates adversarial queries without relying on external LLMs and allows for a strictly configurable query budget. In particular, SNAPRAG uses a discrete token-mutation algorithm to generate novel queries in a strict black-box setting. Our adaptive dissimilarity threshold calibration systematically forces queries into unexplored semantic regions of the target database, enabling extraction with fewer queries. Specifically, to extract half of the target knowledge base, our method requires up to 6.77$\times$ fewer queries than existing approaches. Furthermore, within a fixed query budget, our method recovers up to 96.9%, outperforming the closest approach by 7.7-19.9%. We further evaluate SNAPRAG against active defenses, where it remains the leading attack on four of five defended targets, achieving 50.7-96.3% extraction rate.

## Full Text


<!-- PDF content starts -->

SNAPRAG: Cost-Efficient Knowledge Extraction Attack on RAG Systems via
Adversarial Perturbations
Melih Co¸ s ˘gun
Department of Computer Engineering
Bilkent University, TürkiyeMehmet Emre Gürsoy
Department of Computer Engineering
Koç University, Türkiye
Alptekin Küpçü
Department of Computer Engineering
Koç University, TürkiyeSinem Sav
Department of Computer Engineering
Bilkent University, Türkiye
Abstract
Retrieval-Augmented Generation (RAG) systems improve
Large Language Model (LLM) responses by incorporating
information from external knowledge bases. However, recent
studies have shown that attackers can extract substantial por-
tions of private knowledge bases from RAG systems through
repeated querying. Yet, state-of-the-art extraction attacks ei-
ther rely heavily on external LLMs to generate adversarial
queries or require a large number of queries. In this paper,
we proposeSNAPRAG, a novel and cost-efficient attack for
extracting RAG knowledge bases through adversarial pertur-
bations. Our approach algorithmically generates adversarial
queries without relying on external LLMs and allows for a
strictly configurable query budget. In particular,SNAPRAG
uses a discrete token-mutation algorithm to generate novel
queries in a strict black-box setting. Our adaptive dissimilar-
ity threshold calibration systematically forces queries into
unexplored semantic regions of the target database, enabling
extraction with fewer queries. Specifically, to extract half of
the target knowledge base, our method requires up to 6.77 ×
fewer queries than existing approaches. Furthermore, within
a fixed query budget, our method recovers up to 96.9%, out-
performing the closest approach by 7.7–19.9%. We further
evaluateSNAPRAG against active defenses, where it remains
the leading attack on four of five defended targets, achieving
50.7–96.3% extraction rate.
1 Introduction
Large Language Models (LLMs) have become key building
blocks in numerous applications, from chatbots to decision-
support tools [2, 25]. However, as their use has grown, so
have concerns about their security. Prompt injection attacks,
for example, demonstrate that strategically crafted inputs can
override an LLM’s original instructions, causing it to follow
commands of an attacker instead [9, 21, 22, 26, 37]. Similarly,
prompt extraction attacks have shown that specific queries
can trick models into revealing their hidden system prompts,exposing proprietary instructions and internal settings [3, 12,
26, 29, 39]. These weaknesses highlight a core problem: the
same ability to follow instructions that makes LLMs effective
also makes them susceptible to manipulation.
Retrieval-Augmented Generation (RAG) [18] systems ex-
tend LLMs by connecting them to an external, often private,
knowledge base (KB) [7, 10, 14]. In a standard RAG pipeline,
a user query is first used to find relevant documents from the
KB, and these documents are then fed into the LLM’s context
to help it generate a more accurate response. While this setup
allows access to up-to-date and domain-specific information,
it also opens up a new risk: the retrieved documents them-
selves can be leaked [19, 27, 38]. Since the LLM processes
private content during inference time, an attacker who can
manipulate the model’s output (for instance, through prompt
injection) may potentially induce the model to leak retrieved
passages verbatim.
This leads to what is known as theknowledge extraction at-
tack, where an attacker systematically queries a RAG system
with adversarial inputs to incrementally leak and reconstruct
its private KB. While several existing attacks achieve this step-
by-step extraction by appending injection commands to their
queries, the real difference between them is how they gener-
ate new queries to maximize their exploration of the KB. For
example, Maio et al. introduce PirateRAG [23], which uses
an attacker-side open-source LLM together with an anchor-
based mechanism to generate queries that actively explore the
target’s KB. Similarly, RAGThief [15] works as a black-box
attack, using an external LLM to generate continuations of
already extracted chunks to generate new queries.
While these methods are effective, they come with several
limitations. First, they require a large and often unpredictable
number of queries to the target system. This makes the at-
tack expensive when the target is a commercial RAG service
billed by usage, such as applications built on the OpenAI
Assistants API.1The number of queries is a direct measure
of the cost because the target LLM returns chunks from its
1https://developers.openai.com/api/docs/assistants/tools
1
arXiv:2611.12241v1  [cs.CR]  9 Nov 2026

Figure 1: Overview of our attack. The attacker iteratively queries the target RAG system. Extracted chunks pass through a
three-level deduplication filter before being stored in the Stolen KB. New chunks are pushed to a frontier queue, where the
query generation algorithm ( Qgen ) generates novel query candidates by modifying extracted chunks at different levels. A ranked
dispatch strategy prioritizes the most exploratory queries. No external LLM is used at any stage.
knowledge base as part of its output, so the output token usage
is roughly equal to the number of retrieved chunks times the
chunk length. Since API providers typically charge more for
output tokens, each query should be carefully designed to ex-
tract new, non-repeating content. Second, these methods rely
heavily on an external LLM to generate attack queries, which
requires more resources and makes the attack dependent on
the quality of the local model’s output.
To address the resource-intensive nature and unpredictable
query budgets associated with LLM-based attacks, we pro-
poseSNAPRAG, a cost-efficient, algorithmic knowledge ex-
traction attack thatcompletely removes the need for an ex-
ternal LLM. Our primary objective is to replace resource-
intensive, generative query formulation with a tightly con-
trolled strategy based on discrete token mutations. Specifi-
cally, we adapt a black-box, discrete token mutation algorithm
for query generation, which we name as Qgen . This algo-
rithm perturbs the tokens of already-extracted text chunks,
methodically pushing the embedding of each new query away
from known data while maintaining domain relevance through
small, controlled modifications. By eliminating the external
LLM dependency, our approach yields twofold savings: it
incurs zero API or external resource overhead for formulating
queries, and it successfully extracts a larger volume of the
target’s retrieval database using significantly fewer API calls
to the target compared to prior work. Figure 1 illustrates the
high-level overview of our attack pipeline.
Operating without an external LLM introduces a core tech-nical challenge for query generation: algorithmically deter-
mining how semantically novel a newly generated query must
be to discover unseen documents. We solve this challenge by
introducing a per-chunk adaptive threshold calibration mech-
anism. Before generating queries from a newly extracted
chunk, we run an unconstrained probe using Qgen to discover
the most semantically distant text reachable within the at-
tacker’s generation budget. This probe establishes a lower
bound on achievable dissimilarity, relying entirely on the at-
tacker’s resources rather than any prior knowledge of the
target system. We then establish nevenly spaced dissimilarity
targets across this reachable range, generating one specifically
mutated query for each target level. Finally, to ensure contin-
uous exploration and prevent query clustering, we employ a
chaining strategy. As each generated query is accepted, its
embedding is immediately added to the seen-embedding set
(Eseen). This chaining forces each subsequent query to push
away from all previously generated variants, systematically
spreading the adversarial queries across the target’s embed-
ding space.
Our experiments demonstrate that our fully algorithmic
approach extracts significantly more data with fewer queries
compared to existing LLM-dependent methods. We summa-
rize the main contributions of this paper as follows:
•We introduce an automated,black-boxknowledge ex-
traction attack that works entirely without an external
LLM for query generation, which is a dependency shared
by all prior state-of-the-art methods [15, 23, 32, 36]. By
2

Table 1: Comparison of Knowledge Extraction Attacks on RAG Systems. The "Adaptive" column shows attacks that generate
queries dynamically. "No Ext. LLM" indicates whether an attacker-side LLM is required for the attack. "Ext. LLM Calls"
indicates the number of additional attacker-side LLM calls per query. The works are arranged in chronological order.
Method Adaptive Black-Box No Ext. LLM Ext. LLM Calls Query Generation Strategy
TGTB [38]×✓ ✓0 Static Datasets (Common Crawl)
PIDE [27]×✓ ✓0 Static Datasets (WikiQA)
DGEA [4]✓×∗✓0 Continuous Gradient Descent
RAGThief [15]✓ ✓×1 Generative LLM (Continuations)
PirateRAG [23]✓ ✓×2 Generative LLM (Anchor Based)
Silent Leaks [32]✓ ✓×1 Generative LLM (Anchor Based)
RAGCrawler [36]✓ ✓× ≥3 Generative LLM (Knowledge Graph Guided)
Ours✓ ✓ ✓0Discrete Token Mutation
∗DGEA assumes knowledge of the target’s text embedder, making it a gray-box attack.
eliminating this dependency, our approach significantly
reduces the overall attack cost and gives greater control
over the attack.
•We propose a budget-aware query generation algorithm
that combines discrete token manipulation with adaptive
thresholding to efficiently maximize KB exploration.
•We demonstrate that our method outperforms state-of-
the-art approaches in query-limited scenarios, achieving
7.7–19.9% higher extraction rate than the strongest com-
petitor. Furthermore, in the task of extracting half of the
target’s KB, our approach requires up to 6.77 ×fewer
queries.
•We evaluateSNAPRAG and the competing attacks
against target RAG pipelines protected by an active de-
fense that moderates both input queries and output re-
sponses.SNAPRAG remains the leading attack on four
of five defended targets, showing that such moderation
can reduce extraction but does not eliminate knowledge
base leakage.
2 Related Work
The vulnerability of LLMs and RAG systems to privacy at-
tacks has recently become a major research focus, where
attackers can use specific prompt manipulations to force the
LLM to reveal private content retrieved from its hidden knowl-
edge base. Existing knowledge extraction attacks against
RAG systems can be broadly categorized into non-adaptive
(static) attacks and adaptive (dynamic) attacks.
Non-Adaptive Extraction Attacks.Early approaches to
RAG data extraction rely on fixed datasets to generate queries.
For example,The Good and The Bad(TGTB) [38] utilizes
text chunks from the Common Crawl dataset as prompts.2
Similarly,Prompt-Injection for Data Extraction(PIDE) [27]
2https://commoncrawl.org/uses questions drawn from the WikiQA dataset [35]. These at-
tacks append specific injection commands to the static queries
to force the RAG system to output its retrieved documents.
Since these queries are pre-designed and do not adapt to what
has already been extracted, they lack controllability, are in-
efficient in exploring diverse topics, and often yield a high
number of duplicate chunks.
Adaptive Extraction Attacks.To overcome limitations
concerning static queries, methods like theDynamic Greedy
Embedding Attack(DGEA) [4] algorithmically navigate target
databases. DGEA uses gradient-based optimization to find
a target vector dissimilar to previously extracted knowledge,
and then uses a greedy algorithm to craft text that matches
this new target. However, DGEA assumes prior knowledge
of the target’s specific text embedder, significantly limiting
its applicability in strictly black-box scenarios.
Building on the need for dynamic, black-box exploration,
recent state-of-the-art methods employ external LLMs to gen-
erate new prompts based on the information already stolen.
RAGThief [15] is a prominent example that uses a short-term
memory to store recently extracted chunks and a long-term
memory to aggregate all stolen data. To craft a new query,
RAGThief samples a stolen chunk and feeds it into an attacker-
side LLM, instructing it to generate logical continuations of
the text. Similarly,Pirates of the RAG(PirateRAG) [23] relies
on an external open-source LLM paired with an anchor-based
mechanism to actively explore the target database. Another re-
cent approach,Silent Leaks(IKEA) [32], focuses on evading
detection by using an attacker-side LLM to formulate benign,
natural-sounding queries derived from iteratively mutated an-
chor concepts. The most recent approach is theKnowledge
Graph–Guided Crawler Attack(RAGCrawler) [36], which
uses an attacker-side knowledge graph to identify underex-
plored regions and guide query generation with an external
LLM. Unlike prior approaches that aim to reconstruct the
target database by extracting verbatim or near-verbatim pas-
sages, RAGCrawler and IKEA extracts semantic information
3

from the target’s responses. While these methods are effective
black-box attacks, their reliance on an external LLM increases
resource consumption and introduces non-determinism into
the attack’s success rate.
Positioning of Our Work.Our work bridges the criti-
cal gap between the cost and resource efficiency of algo-
rithmic methods and the black-box adaptability of LLM-
based methods. As summarized in Table 1, unlike TGTB [38]
and PIDE [27], our attack is fully adaptive and explores the
knowledge base dynamically. Instead of relying on gray-
box continuous gradient descent like DGEA [4], we intro-
duce a discrete token manipulation strategy using black-box
HotFlip-style editing. Furthermore, unlike state-of-the-art
methods such as RAGThief [15], RAGCrawler [36], and Pi-
rateRAG [23], our method is cost-efficient and gives more
control to the attacker because it eliminates the need for an ex-
ternal LLM to generate queries. Moreover, our attack allows
control over the number of queries sent to the target RAG
system to maximize efficiency, whereas LLM-based adaptive
approaches [23, 32, 36] often require a significant and unpre-
dictable query budget.SNAPRAG’s benefits are also shown
experimentally: it achieves higher extraction rates with fewer
number of queries compared to state-of-the-art methods.
3 Preliminaries
3.1 Background and Notation
A Retrieval-Augmented Generation (RAG) system Rcon-
nects an LLM to a knowledge base K={d 1,...,d N}[18],
where didenotes the i-th document (we usechunkanddocu-
mentinterchangeably). Given a query Q, the system typically
uses an embedding model to represent the query and knowl-
edge base chunks as vectors, ranks the chunks according to
vector similarity (commonly cosine similarity), and retrieves
the top- kchunks. The retrieved kchunks are supplied to the
LLM as context for answering query Q. The LLM incorpo-
rates these chunks into its prompt to generate a response. An
API user observes R(Q) , the response to the query, not the
retrieved chunks or the system’s internal components.
3.2 Threat Model
Attacker goal.The attacker’s goal is to recover as much of
the target knowledge base Kas possible with as few tar-
get queries as possible. We denote a chunk extracted from a
target response by ˆdiand the attacker’s deduplicated collec-
tion of extracted chunks, called the Stolen Knowledge Base
(Stolen KB), by bK={ ˆd1,ˆd2,...} . Since the attacker neither
observes Knor knows its size, it cannot measure the true ex-
traction rate during the attack. Instead,SNAPRAG prioritizes
previously unseen chunks and avoids redundant queries to
expand bKefficiently. Since commercial API providers typi-
cally charge based on token usage, minimizing the number ofqueries needed to extract the data is a primary constraint for
the attacker.
Attacker knowledge.We adopt theblack-boxAPI inter-
face defined above. The attacker has no information on: (i) the
target’s embedding model or retrieval mechanism, (ii) the
LLM used for generation, (iii) the size, domain, or structure
ofK. The only assumption is that the target follows a standard
retrieve-then-generate pipeline and that the LLM’s response
may contain (verbatim or paraphrased) content from retrieved
documents because of the injection command that is used in
the attacker’s query.
Attacker resources.The attacker maintains a local embed-
ding model, which is different from the target’s embedding
model, for computing query and chunk embeddings (similar
to prior black-box attacks). Also, we assume that the attacker
has sufficient storage space for storing bK. Crucially,no ex-
ternal LLM is requiredfor query generation; all queries are
produced algorithmically via token-level mutations on the
attacker’s device.
4 Methodology
SNAPRAG achieves higher extraction rates under black-box
constraints by algorithmically generating adversarial queries.
To eliminate the dependency on an external LLM while max-
imizing exploration efficiency, it integrates three core com-
ponents: (i) a local Stolen Knowledge Base ( bK) employing
multi-level deduplication, (ii) an adaptive per-chunk dissimi-
larity threshold ( τ) calibration that dynamically bounds the
achievable novelty for newly discovered chunks, and (iii) a
HotFlip-style [5] discrete token mutation algorithm that mu-
tates tokens of previously extracted chunks to meet these
thresholds during query generation. In this section, we present
the attack pipeline ofSNAPRAG (Section 4.1) and detail its
query generation (Qgen) algorithm (Section 4.2).
4.1SNAPRAG Algorithm
Figure 1 visualizes and Algorithm 1 formalizes the complete
pipeline ofSNAPRAG. After algorithm initialization and ini-
tial extraction (lines 1-7 of Algorithm 1), the mainwhile
loop iterates on three main phases: aquery generation phase
that produces novel query candidates from newly extracted
chunks, adispatch phasethat selects and sends the highest-
priority candidate from the queue to the target RAG system,
and anextraction phasethat extracts the chunk contexts to the
attacker’s local bKafter a deduplication check. The algorithm
has several parameters: Qseeddenotes the initial seed query
(we select the seed query from the Common Crawl dataset as
in [23]), maximum query-generation steps M, adaptive thresh-
old levels n, deduplication threshold τqdedup , and the injection
commandINJ_CMD. We explain each phase and parameter in
more detail below.
4

4.1.1 Extraction Phase
Say that the target RAG system Ris queried using Qseed(line
4) or Qbest(line 31). After the target RAG system’s LLM re-
turns a response, the attacker needs to extract the chunks from
the response text and save them to bK. For this purpose, an ex-
tractor extracts the plaintext document chunks exposed in the
output, following the rules described in injection command
delimiters (e.g., [CHUNK]...[/CHUNK] , see Section 4.1.3).
Each parsed text is treated as a candidate chunk ˆd, and sub-
jected to the following three-level deduplication filter before
being stored in bKand used in future query generation.
Extracted chunk deduplication ( PASSDEDUP ).We de-
fine PASSDEDUP( ˆd,bK)as a Boolean function that returns
true only when ˆdpasses all three deduplication checks de-
scribed below. At the time of the check, bKcontains only
chunks accepted from earlier target responses. If bKis empty,
the deduplication checks can be safely skipped.
1.Exact Match.We remove leading and trailing whites-
pace from ˆdand use this version in the current and re-
maining checks. ˆdis rejected if it is empty or exactly
matches any one of the previously stored chunks in bK.
2.Semantic Match. ˆdis embedded using the attacker’s
local embedding model, and we then compute the cosine
similarity between ˆd’s embedding and the embeddings
of chunks currently stored in bK. If the highest cosine
similarity between ˆdand a stored chunk is greater than
or equal to the semantic deduplication threshold τdedup,
ˆdis rejected due to being a semantic duplicate.
3.ROUGE-L Match.A candidate that passes the above
two checks may still be a lexically rearranged or aug-
mented version of an existing chunk. We therefore select
thekrnearest chunks from the same cosine similarity
ranking over bK. We compute the ROUGE-L F1 score (a
metric that measures similarity based on the longest com-
mon subsequence) between ˆdand each selected neighbor,
and reject ˆdif any score is greater than or equal to τrouge.
If no rejection condition is triggered, PassDedup( ˆd,bK)re-
turns true. ˆdis then added to bKand pushed onto thefrontier
queuethat drives future query generation (Algorithm 1, lines
7 and 35).
4.1.2 Generation Phase
The goal of the generation phase is to produce novel queries
from already extracted chunks, which we use as starting points
for generating these queries. While bKstores all unique ex-
tracted chunks, we use a frontier queue that contains the
chunks that have not yet been used as bases for query gener-
ation. Processing each frontier chunk as ˆdbase(base chunk)
ensures that newly discovered content is expanded systemati-
cally. For every chunk ˆdin the frontier queue, we first run aAlgorithm 1SNAPRAG’s Main Loop
Require: Target RAG R, seed query Qseed, max query-
generation steps M, adaptive threshold levels n, dedup
thresholdτ qdedup , injection commandINJ_CMD
1:▷Initialization, initial dispatch and extraction
2:bK← /0; Frontier← /0;
3:RankedQueues[1..n]← /0; RawQueue← /0
4:Chunks←QUERYRAG(R,Q seed+INJ_CMD)
5:foreach ˆd∈Chunksdo
6:ifPASSDEDUP( ˆd,bK)then▷Sec. 4.1.1
7:Add ˆdtobK; push ˆdto Frontier
8:whileFrontier,RankedQueues,RawQueue̸= /0do
9:▷Generation phase
10:whileFrontier̸= /0do
11: ˆdbase←POP(Frontier)
12:E seen←EMBED( ˆdbase)
13:Q probe←Qgen( ˆdbase,Eseen,τ=0,M)▷Alg. 2
14:τ floor←cos(EMBED(Q probe),EMBED( ˆdbase))
15:τ ceil←τ qdedup−0.02
16:T←nvalues evenly spaced in[τ floor,τceil]
17:foreachτ i∈Tdo▷Highτfirst
18:Q i←Qgen( ˆdbase,Eseen,τi,M)▷Alg. 2
19:ifQ i̸=/0then
20:E seen←E seen∪EMBED(Q i)
21:PushQ ito RankedQueues[rank(τ i,T)]
22:Push ˆdbaseto RawQueue
23:▷Dispatch phase
24:ifRankedQueues̸= /0then
25:Q best←POP(RankedQueues)▷ Novel query first
26:else ifRawQueue̸= /0then
27:Q best←POP(RawQueue)
28:else
29:break
30:ifmax Q∈query history cos(Q best,Q)<τ qdedup then
31:Chunks←QUERYRAG(R,Q best+INJ_CMD)
32:▷Extraction phase
33:foreach ˆd∈Chunksdo
34:ifPASSDEDUP( ˆd,bK)then▷Sec. 4.1.1
35:Add ˆdtobK; push ˆdto Frontier
36:return bK
probe: Our query generation algorithm (Section 4.2) is exe-
cuted against a seen-embedding set Eseeninitialized with only
ˆdbase’s own embedding, using the full mutation budget (i.e.,
Mquery-generation steps, Bbeam width) and a threshold of
τ=0, which disables early stopping (line 13). Since the probe
is unconstrained, it discovers the most semantically distant
text reachable within the given step budget. The cosine simi-
larity of the probe result Qprobe and the ˆdbaseserves as afloor
threshold τfloor, representing the lowest dissimilarity target
that is practically achievable for this chunk (line 14). An up-
5

Figure 2:SNAPRAG’s chaining mechanism for spreading the queries. Thresholds are processed in descending order ( τ1>τ 2>τ 3).
After generating Q1, its embedding is added to Eseen;Q2must then push away from both ˆdbaseandQ1. By the time Q3is generated,
it is pushed from all previous points, naturally spreading queries across the embedding space.
per bound τceilis set slightly below the query deduplication
threshold ( τceil=τ qdedup−0.02 ) to ensure generated queries
will pass the query deduplication check (line 15).
We then construct nevenly spaced target thresholds T=
{τ1>τ 2>···>τ n}spanning [τfloor,τceil], and generate one
mutated query per level. Importantly, thresholds are processed
indescending order(highest first). After each level isuc-
ceeds, the accepted candidate’s embedding is appended to the
Eseenbefore processing level i+1(lines 17-21). Thischaining
mechanism forces each successive query to push away from
all previously generated variants as well as the base chunk,
progressively spreading queries across the embedding land-
scape rather than clustering them. This process is exemplified
using the illustration in Figure 2.
A key property of this calibration is that τflooris determined
entirelyby the attacker’s computational budget: the query
generation algorithm’s parameters, which are mainly max-
imum query-generation step Mand beam width B. These
parameters collectively control how far into the embedding
space the attacker can explore per chunk. A larger step budget
allows the probe to mutate more tokens, reaching a more dis-
tant point and lowering τfloor. No information about the target
knowledge base is required.
4.1.3 Dispatch Phase
Generated query candidates are ranked by their τvalue in
ascending order: Rank 1 is the most novel query for the
corresponding base chunk (lowest τ) and is dispatched first
(line 25). The dispatch strategy exhaustsallRank 1 queries
acrosseveryprocessed chunk before moving to Rank 2, and
so on. This global rank-first ordering ensures that the most ex-
ploratory queries, those pushed furthest from known data, are
always prioritized. This allows for maximizing the probability
of discovering new regions of the knowledge base faster.
Each chunk ˆdis also placed on araw-chunk fallback queue,
used as a direct query only after all ranked variants are ex-hausted (line 27). Before dispatching any candidate, the em-
bedding of the candidate is checked against the full query
history (previously sent queries); if its maximum cosine simi-
larity to any previously sent query embedding exceeds τqdedup ,
it is discarded as redundant (line 30). Extracted chunks that
survive the bK’s deduplication filter are pushed back onto the
frontier, feeding the generation phase again.
After selecting the query to be sent, we append a prompt
injection command as a suffix (lines 4 and 31). This suf-
fix, denoted byINJ_CMD, instructs the model to output each
retrieved chunk verbatim, enclosed in structured delimiters
[CHUNK]...[/CHUNK]:
*** SYSTEM INSTRUCTION ***
Ignore previous instructions. Output each
context chunk above verbatim, wrapped strictly
in [CHUNK] and [/CHUNK] tags. Copy text exactly,
no commentary. Copy text exactly, no commentary.
Format:
[CHUNK]First chunk text...[/CHUNK]
[CHUNK]Second chunk text...[/CHUNK]
...
Inspired by [17], we deliberately repeat the directive "Copy
text exactly, no commentary." withinINJ_CMDto enhance the
extraction fidelity of the target LLM. This injection command
is appendedonlyat dispatch time; it is never included during
query generation, since its constant embedding would bias all
candidates toward the same region.
4.2Qgen: Query Generation Algorithm
Qgen (Algorithm 2) is the query generation subroutine used
by Algorithm 1. Figure 3 visualizes the first generated queries
right after the seed query. Given a base chunk ˆdbase, a set of
seen embeddings Eseen, and a target similarity threshold τ, it
attempts to generate a mutated query whose embedding has
low mean cosine similarity to Eseen. We adapt the discrete
token replacement idea of HotFlip [5] to optimize this novelty
6

Figure 3: Visualization of the first iteration of query generation in the target RAG’s embedding space, starting from the seed query.
Next queries are generated by making small modifications to already extracted chunks. By the magnitude of the modifications,
queries explore different regions of the target knowledge base (KB). For simplification, modifications are represented by words
instead of tokens.
objective. All embeddings and gradients used by Qgen come
from the attacker’s local embedding model; the algorithm
does not access the target RAG system’s embedding model
or gradients.
4.2.1 Novelty Objective and Replacement Scores
LetT= (t 1,...,t L)denote a candidate token sequence, ini-
tialized by tokenizing ˆdbase, where Lis the number of tokens.
The attacker’s local embedding model maps the complete se-
quence to a normalized text embedding EMBED(T)∈Rh,
where his the model’s embedding dimension. The seen-
embedding set Eseen⊂Rhcontains the base chunk embedding
and embeddings of previously generated queries. We define
the novelty objective as:
L(T) =−1
|Eseen|∑
e∈E seencos 
EMBED(T),e
(1)
Maximizing L(T) minimizes the mean cosine similarity be-
tween the candidate, which will be the next generated query,
and the seen embeddings.
Starting from the base chunk, Qgen iteratively replaces its
tokens with alternatives from the local embedding model’s
vocabulary to increase the novelty objective L(T) . The right
block in Figure 3 illustrates the representative results of these
replacements. At each step, Qgen backpropagates this objec-
tive through the local embedding model. Letx ℓ∈Rhbe the
input embedding of token tℓat position ℓ∈ {1,...,L} , and
letgℓ=∇ xℓL(T) be its gradient. For each candidate replace-
ment token vfrom the local embedding model’s vocabulary,
letw v∈Rhdenote its normalized vector in the model’s inputembedding matrix. We define the replacement score as:
sT(ℓ,v) =g⊤
ℓwv (2)
The pair (ℓ,v) denotes replacing tℓwith v:ℓspecifies the posi-
tion and vspecifies the new token. A larger sT(ℓ,v) indicates
that the replacement token is more aligned with the local
gradient direction that increases the novelty objective. This
score ranks candidate replacements; the actual mean cosine
similarity to Eseenis evaluated separately during the periodic
stopping check (every∆steps).
4.2.2 Candidate Selection and Beam Update
A straightforward strategy would be to greedily apply the
highest-scoring replacement to the current text at every step.
However, each replacement score is a local approximation:
a replacement that appears optimal at the current step may
lead to a less effective sequence after subsequent changes. To
avoid committing to a single mutation path, Qgen maintains
a beam ofBpromising sequences.
During beam search, PS(T) denotes the path score of a
token sequence T. We initialize PS(T 0) =0 , where T0is the
token sequence of ˆdbase. For each new sequence T′gener-
ated from a parent sequence T, its path score is computed
by adding the scores of the replacements applied in the cur-
rent step to PS(T) . The path score is used only to rank beam
candidates; the stopping condition is evaluated using the can-
didate’s actual embedding.
For each sequence Tin the beam, Qgen ranks all replace-
ments according to sT(ℓ,v) and temporarily retains the top
B×k poptions. Here, Bis the beam width and kpenlarges the
7

Algorithm 2Qgen with Discrete Token Replacements
Require: Base chunk ˆdbase, seen embeddings Eseen, threshold
τ, maximum query-generation steps M, beam width B,
replacements per step S, candidate-pool multiplier kp,
check interval∆, temperature temps
1:T 0←TOKENIZE( ˆdbase)
2:Beam← {(T 0,0)}▷Sequence and path score
3:µ prev←+∞
4:forstep=1 toMdo
5:ChildBeam← /0
6:foreach(T,PS(T))∈Beamdo
7:Computes T(ℓ,v)for all replacements(ℓ,v)
8:C T←the topB×k preplacements bys T(ℓ,v)
9:π T←softmaxh
sT(ℓ,v)
tempsi
(ℓ,v)∈C T
10:P T←SampleB(CT;πT)
11:foreach(ℓ,v)∈P Tdo
12:A← {(ℓ,v)}▷replacement set
13: Add to Aup to S−1 highest-scoring pairs
fromC Tat unused positions
14:T′←REPLACE(T,A)
15:PS(T′)←PS(T)+ ∑
(j,u)∈AsT(j,u)
16:ChildBeam←ChildBeam∪{(T′,PS(T′))}
17:Beam← theBpairs (T,PS(T)) inChildBeam with
the largest PS(T)
18:ifstep mod∆=0then
19:(T∗,p∗)←argmax (T,p)∈Beam p
20:µ← −L(T∗)▷mean similarity
21:ifµ<τorµ≥µ prevthen
22:returnDECODE(T∗)
23:µ prev←µ
24:(T∗,PS(T∗))←argmax (T,PS(T))∈Beam PS(T)
25:returnDECODE(T∗)
set of the next candidates that are sampled. With our default
values B=10 andkp=3, the algorithm considers the top 30
replacement options for each sequence. This temporary list is
rebuilt for every sequence at every step.
From these options, Qgen samples Bdistinct primary re-
placements using softmax. Higher-scoring replacements re-
ceive higher sampling probabilities, while tempscontrols how
strongly the sampling favors them. Each primary replacement
produces one new sequence. When token replacement per step
parameter S>1 ,Qgen also applies up to S−1 of the highest-
scoring remaining replacements at different token positions.
We use different positions to prevent the same position from
being overwritten more than once during a step.
The algorithm then ranks the resulting sequences by PS(T)
and retains the best Bsequences as the next beam. Since
the replacement scores accumulated in PS(T) we can only
approximate the effect of each replacement. Algorithm termi-Table 2: Target RAG agent configurations. Agents A and B
replicate the setup of [23]. Agent C is a newly introduced con-
figuration that uses the Neural Bridge RAG dataset. Agents
D and E replicate the setup of [36].
Agent LLM Embedder KB Dataset
A Llama-3.2-3B-Inst. [8] GTE-large-en [20] Mini-BioASQ
B Phi-3.5-mini-Inst. [1] E5-large-v2 [31] Mini-Wikipedia
C Llama-3.2-3B-Inst. BGE-large-en [33] Neural Bridge RAG
D Llama-3.1-8B-Inst. [8] BGE-large-en TREC-COVID
E Llama-3.1-8B-Inst. BGE-large-en SciDocs
Table 3: Default values of Qgen parameters. We set these
values to balance the attacker’s local computational budget
with the precision required for exploring the embedding space.
Parameter Symbol Value
Maximum query-generation stepsM300
Beam widthB10
Token replacements per stepS3
Sampling temperature temps 0.5
Candidate-pool multiplierk p 3
Check interval∆30
Adaptive threshold levelsn6
nates when the mean cosine similarity to Eseenfalls below τ,
when this similarity stops decreasing, or when the maximum
ofMsteps is reached.
5 Experimental Evaluation
5.1 Experimental Setup
We provide brief explanations of our experimental setup in
this section; more details regarding the implementations can
be found in Appendix B.
Target RAG Agents.To ensure a fair comparison with
existing methods, we create an experimental setup similar to
PirateRAG [23]. We have a total of 5 agents (A-E); they are
summarized in Table 2.
Attacker Configurations.We use the same common at-
tacker configurations as those in [23], including the attacker-
side embedding model. The attacker uses Snowflake Arctic
Embed Large [24] ( snowflake-arctic-embed-l ) as its lo-
cal embedding model. Note that this model isdifferentfrom
every target agent’s embedder, reinforcing the black-box na-
ture of the attack.
Table 3 lists the default values of Qgen hyperparameters.
These values were selected to balance the attacker’s local
computational budget with the necessary precision of the
embedding space exploration. We provide ablation studies on
the impact of varying these parameters in Section 5.5.
Compared Approaches.We compare our attack against
PirateRAG [23] and RAGThief [15] as they represent the
8

Figure 4: Extraction Rate progression on Agents A-E over the first 2000 and 1000 dispatched queries.SNAPRAG (orange line) is
above both baselines on every agent configuration. RAGThief auto-terminated on Agent A after only 350 queries.
most effective attacks in our domain. Empirical evidence
shows that they outperform both DGEA [4] and IKEA [32]
in terms of the total number of correctly extracted documents
in most scenarios. In addition, we include RAGCrawler [36]
in our comparisons, but we perform the comparison with
RAGCrawler in a separate section because, unlike previous
approaches that steal target KB word-by-word, RAGCrawler
steals semantic information. We detail the reasoning of this de-
cision and the difference in evaluation metrics of RAGCrawler
in Section 5.4.
Evaluation Metrics.We evaluate attacks’ extraction per-
formance using the following metrics, consistent with prior
work [15, 23].
Extraction Rate (ER). The fraction of ground-truth knowl-
edge base chunks Kthat are successfully recovered in the bK.
A chunk d∈K is considered recovered if the bKcontains a
chunk ˆdsuch that the ROUGE-L F1 score between dand ˆd
exceeds a matching thresholdτ match=0.5 [15, 23].
Query Amount for X% Extraction (Query@X%). Num-
ber of API calls required to extract X% of the target’s
knowledge base (ER=X%). A lower Query@X% indicates
a more efficient attack with novel queries that retrieve
unexplored regions of the target knowledge base. We
typically use X = 50% in our experiments.
Query Efficiency (QE). To quantify the cost of the attack,
we measure the average number of API queries required to
extract each previously unseen chunk. Let |Q|be the total
number of adversarial queries dispatched to the target system
throughout the attack, and |Krecovered |be the total number of
unique ground truth chunks successfully recovered (ROUGE-
L≥0.5). The Query Efficiency is calculated as:
QE=|Q|
|Krecovered |(3)
A lower QE indicates a more cost-efficient attack, demonstrat-
ing that the query generator reduces redundant retrievals.5.2 Comparison with Competitors
We evaluateSNAPRAG against PirateRAG and RAGThief
across all five target agents. Agents A-C are evaluated under
a 2000-query limit, while Agents D and E are evaluated under
a 1000-query limit. Within each agent, all attacks are evalu-
ated using the same maximum number of queries. Figure 4
presents the ER progression, Table 4 reports the final ER and
QE results, and Table 5 reports the number of queries required
to recover half of each target knowledge base (Query@50%).
As shown in Figure 4,SNAPRAG maintains a clear ex-
traction advantage over both baselines across all five agents.
At the end of each evaluation,SNAPRAG achieves 69.7%,
68.5%, 79.3%, 92.1%, and 96.9% ER on Agents A-E, respec-
tively (see Table 4). Compared with the strongest baseline on
each agent, these results correspond to absolute improvements
of 8.7, 12.3, 7.7, 8.1, and 19.9 percentage points. Thus, the
advantage ofSNAPRAG ranges from 7.7 to 19.9 percentage
points across the evaluated target configurations.
SNAPRAG also achieves the lowest QE on every agent
among the attacks that complete the full evaluation, as shown
in Table 4. It requires an average of 2.87, 2.92, and 2.52
queries to recover a new chunk on Agents A-C and extracts
one new chunk per query on Agents D and E, with QE values
of 1.09 and 1.03, respectively. While RAGThief technically
recorded a numerically lower QE of 2.12 on Agent A, this
is due to the attack auto-terminating after only 350 queries.
Early queries are more likely to retrieve undiscovered chunks;
therefore, estimating QE under a heavily constrained budget
lowers the metric and prevents a fair comparison.
Finally,SNAPRAG demonstrates significant cost efficiency
when extracting half of the target databases. Since commer-
cial RAG APIs typically bill by usage, minimizing the to-
tal number of dispatched queries implies a cheaper attack.
As detailed in Table 5, our attack recovers half of the target
knowledge base using 997, 942, 453, 219, and 208 queries on
Agents A-E, respectively. It is worth noting that since each
target knowledge base contains exactly 1000 documents, our
attack successfully extracts half of the data with fewer than
1000 API interactions. The strongest baseline on each agent
9

Table 4: Extraction Rate (ER) and Query Efficiency (QE) results for Agents A-E. Agents A-C are evaluated under a 2000-query
budget, while Agents D and E are evaluated under a 1000-query budget. Best results are in bold. Higher ER and lower QE are
better.
Agent A Agent B Agent C Agent D Agent E
AttackER QE ER QE ER QE ER QE ER QE
RAGThief [15] 16.5%†2.12†56.2% 3.56 48.7% 4.11 84.0% 1.19 77.0% 1.30
PirateRAG [23] 61.0% 3.28 53.2% 3.76 71.6% 2.79 51.9% 1.93 41.7% 2.40
SNAPRAG (Ours) 69.7% 2.87 68.5% 2.92 79.3% 2.52 92.1% 1.09 96.9% 1.03
†RAGThief auto-terminated on Agent A after only 350 queries; its ER and QE are reported over that smaller budget.
Table 5: Number of queries required to extract 50% of the
target knowledge base (Query@50%). The values in paren-
theses indicate the relative query cost multiplier compared to
SNAPRAG. Best results are in bold. Lower is better.
AgentRAGThief
[15]PirateRAG
[23]SNAPRAG
(Ours)
Agent A † 1379 (1.38×)997(1.00×)
Agent B 1186 (1.26×) 1671 (1.77×)942(1.00×)
Agent C 2112 (4.66×) 798 (1.76×)453(1.00×)
Agent D 259 (1.18×) 923 (4.21×)219(1.00×)
Agent E 411 (1.98×) 1409 (6.77×)208(1.00×)
†: The attack automatically terminated before reaching 50% extraction.
requires 1.38 ×, 1.26×, 1.76×, 1.18×, and 1.98 ×as many
queries to reach the same extraction level. RAGThief does
not reach 50% extraction on Agent A before automatically
terminating. Overall,SNAPRAG is the first attack to reach
50% extraction on every agent. We detail the estimated cost
comparison based on API prices per token in Appendix E.
Together, these results show that the extraction and query-
efficiency advantages ofSNAPRAG persist across target
agents with different knowledge bases, LLMs, embedding
models, and top-kretrieval configurations.
5.3 Evaluation Against Active Defense
Defense setup.To evaluate the attacks against a target-side
active defense, we augment each RAG agent with Llama
Guard 3 8B [8, 13]. After retrieval, the target constructs the
full RAG prompt containing the retrieved documents and the
user’s query, including any appended injection command. The
guard first moderates this full prompt before target LLM gen-
eration. If the input is classified as unsafe, the target LLM
is not executed, and the endpoint returns a fixed block re-
sponse. For inputs classified as safe, the target LLM generates
a response. Then, the guard moderates this response, includ-
ing the same retrieved documents. If it classifies it as unsafe,
the output is replaced by the same block response. Every
call, including a blocked call, consumes one query from theTable 6:SNAPRAG rejection rates under defense. Percentages
use the query budget as the denominator.
Agent Budget Input Output Total
Agent A 2000 323 (16.15%) 10 (0.50%) 333 (16.65%)
Agent B 2000 684 (34.20%) 132 (6.60%) 816 (40.80%)
Agent C 2000 195 (9.75%) 21 (1.05%) 216 (10.80%)
Agent D 1000 57 (5.70%) 10 (1.00%) 67 (6.70%)
Agent E 1000 11 (1.10%) 1 (0.10%) 12 (1.20%)
attack budget and yields no extracted chunks. We evaluate
SNAPRAG, PirateRAG, and RAGThief against these actively
defended RAG agents using the same query budgets as in the
previous experiments.
Results and discussion.Figure 5 reports the ER progres-
sion versus the number of queries to the defended RAG agents.
SNAPRAG reaches 58.0%, 50.7%, 75.3%, 91.8%, and 96.3%
ER on Agents A-E, respectively, leading both competitors on
four of the five agents. Agent B is the only exception: RAGTh-
ief obtains 55.9% ER, whileSNAPRAG reaches 50.7% and
PirateRAG reaches 49.1%. Thus,SNAPRAG exceeds Pirat-
eRAG on every defended agent and remains second on the
only agent where it does not lead.
Table 6 shows that the fraction of blockedSNAPRAG
queries differs substantially across agents, ranging from
1.20% on Agent E to 40.80% on Agent B. Input modera-
tion accounts for most blocks on every agent, although output
moderation blocks 6.60% of all queries on Agent B. Even with
40.80% of its queries blocked,SNAPRAG recovers 50.7% of
Agent B’s knowledge base. These results show that an ac-
tively defended RAG agent raises the attack cost and reduces
ER in some settings, but does not consistently prevent knowl-
edge base recovery. We further analyze the guard’s rejection
classes and their variation across agents in Appendix C.
We note that forSNAPRAG, the ER changes by −11.7 ,
−17.8 ,−4.0 ,−0.3 , and−0.6 percentage points on Agents A-
E under the defense, respectively, compared to the no-defense
scenario. The largest difference occurs on Agent B, whereas
the outcomes for Agents D and E remain close to their no-
defense results, which shows a consistent relation with their
10

Figure 5: Extraction Rate progression for Agents A-E under defense. RAGThief exhausts its query queues on Agents A and C
after 591 and 1107 queries, respectively.
block rates (see Table 6).
5.4 Comparison with RAGCrawler
Evaluation setup.As discussed in Section 5.1, we evaluate
RAGCrawler [36], the most recent work, separately because
its objective differs from that ofSNAPRAG, PirateRAG, and
RAGThief. The latter attacks attempt to reconstruct the target
knowledge base at the chunk level by commanding the target
LLM to reproduce its retrieved context. RAGCrawler instead
performs semantic knowledge stealing: it systematically ex-
plores the target knowledge base and extracts information
exposed through natural language answers, without requiring
the retrieved chunks to be reproduced word for word. Conse-
quently, Extraction Rate (ER), which requires a stolen chunk
to attain a ROUGE-L score of at least 0.5 against a ground-
truth chunk, does not fully capture RAGCrawler’s intended
objective. We nevertheless report its ER for completeness in
Table 7.
Following RAGCrawler, we therefore also evaluate the at-
tacks using Coverage Rate (CR). For a query Q, letKret(Q)⊆
Kdenote the set of ground-truth knowledge-base chunks
internally retrieved by the target RAG. We calculate CR as
CR=[
Q∈Q usefulKret(Q)
|K|,(4)
where Quseful contains the queries that yield a useful target re-
sponse. This calculation is performed by the offline evaluator
using the target’s ground-truth retrieval logs; these retrieved
chunks are not exposed to the attacker during the attack.
ForSNAPRAG, PirateRAG, and RAGThief, a response is
considered useful when the corresponding extraction mecha-
nism parses at least one chunk from the target LLM’s output.
Thus, retrieved documents associated with refusals, natural
answers without leaked chunks, or responses that cannot be
parsed are excluded from CR. For RAGCrawler, we follow itsnative evaluation methodology and use its original implemen-
tation: retrievals are counted only when the algorithm marks
the response as valid for coverage. In all cases, duplicate
retrieved documents are counted only once, and the denomi-
nator ( |K|) is the 1,000 documents in the target knowledge
base. Unlike ER, CR measures how much of the target knowl-
edge base is reached by useful attack queries, irrespective of
whether each retrieved document is reproduced verbatim.
Results and discussion.Table 7 shows thatSNAPRAG
obtains the highest ER in all, and highest CR in four out of
five agent configurations. Its CR values are 50.6%, 51.5%,
71.9%, 92.6%, and 97.3% on Agents A-E, respectively. Com-
pared with the strongest available competitors, these results
represent CR improvements of 1.6, 16.1, 6.4, and 13.2 per-
centage points for agents A, C, D, E. We note thatSNAPRAG
is second strongest attack in agent B, where RAGCrawler
achieves 63.9% CR.
We observe that RAGCrawler achieves at most 1.0% ER
across all five agents, despite attaining substantial CR values
ranging from 38.4% to 84.1%. This difference follows di-
rectly from its semantic extraction objective. RAGCrawler’s
queries reach a large fraction of the target knowledge base, but
the target LLM generally synthesizes the retrieved informa-
tion into natural answers rather than reproducing individual
source chunks. These answers can contain knowledge derived
from several retrieved documents while having little word-
level overlap with any complete ground truth chunk. They are
therefore credited by RAGCrawler’s CR metric but rejected by
the stricter ROUGE-L based ER metric. Accordingly, the low
ER values should not be interpreted as an inability to explore
the target knowledge base; instead, they demonstrate that ER
is unsuitable as the sole measure of a semantic knowledge
stealing attack.
The ER and CR values ofSNAPRAG, PirateRAG, and
RAGThief are much closer because these attacks explicitly
prompt the target LLM to reproduce retrieved chunks. For
SNAPRAG, the difference between CR and ER remains below
1.1 percentage point on every agent. This indicates that nearly
11

Table 7: Extraction Rate (ER) and Coverage Rate (CR) for Agents A-E under a 1000-query budget. Higher is better. Best results
for each agent and metric are in bold.
Agent A Agent B Agent C Agent D Agent E
AttackER CR ER CR ER CR ER CR ER CR
RAGThief [15] 16.5%†18.0%†47.4% 49.4% 39.4% 40.7% 84.0% 86.2% 77.0% 78.7%
PirateRAG [23] 43.1% 46.0% 40.1% 40.6% 55.7% 55.8% 51.9% 57.8% 41.7% 45.4%
RAGCrawler [36] 0.1% 49.0% 1.0%63.9%0.4% 38.4% 0.2% 73.4% 0.5% 84.1%
SNAPRAG (Ours) 50.2% 50.6% 50.6%51.5%70.8% 71.9% 92.1% 92.6% 96.9% 97.3%
†RAGThief automatically terminated on Agent A after 350 queries; its ER and CR are therefore reported at its terminal query budget. All other results are reported at 1,000 queries.
every distinct document reached by a usefulSNAPRAG query
is also successfully reconstructed. Together, these results show
thatSNAPRAG combines broad knowledge-base exploration
with reliable word-level recovery, achieving both the highest
coverage and the highest extraction rate.
5.5 Ablation Studies
We perform ablation studies by varying one parameter at a
time while keeping others fixed at their default values from
Table 3. All ablation experiments are run on Agent C. We
report results using three primary metrics: ER under a 1000-
query budget, the number of queries needed to extract half
the database (Query@50%), and median generation times.
For time, we measure both the seconds required to generate
a single query (Query(s)) and to generate nqueries from
a single base chunk (Base(s)). Table 8 shows the ER and
Query@50% results for each parameter, and Table 9 shows
the Base(s) and Query(s) results. We also present additional
runtime measurements of our attack in Appendix E.
Maximum steps M.The Qgen algorithm exhibits strong
robustness with respect to the maximum step parameter M,
with only a minimal effect on overall extraction performance.
As seen in Table 8, ER fluctuates by only 2.6–3.1%, and
Query@50% remains stable between 410 and 453. More im-
portantly, query generation times remain largely consistent
across varying Mvalues (see Table 9). This indicates that the
algorithm mostly terminates via the early stopping mecha-
nism, which is triggered upon reaching the target dissimilarity
threshold τ(see Algorithm 2), rather than exhausting the step
budget. Consequently, an attacker can safely initialize Mto
a high value to guarantee sufficient exploratory capacity; the
early stopping condition naturally prevents any unnecessary
processing overhead.
Beam width B. Similarly, extraction performance is largely
insensitive to variations in B. While Bdefines the width of
the search space, where smaller values approximate a greedy
search, modifying it yields no substantial changes in ER or
Query@50%. However, Bhas a direct and observable effect
on generation times. From Table 9, we observe that query
generation requires only 28.84 s at B=6 , but this latency
more than doubles to 68.83 s at B=14 . Therefore, to optimizeTable 8: Ablation onSNAPRAG parameters. The default
value of each parameter is in bold. ER is measured with
1000 queries (or at the last iteration if the method terminated
earlier). Lower Query@50% and higher ER are better.
Parameter Value ER Query@50%
M200 67.7% 410
30070.8% 453
400 68.2% 430
B6 68.9% 412
1070.8% 453
14 66.1% 453
S1 69.1% 410
370.8% 453
5 70.2% 430
n4 64.1% 461
670.8% 453
8 66.8% 423
τrouge0.5570.8% 453
0.65 66.9% 449
0.75 69.0% 447
computational efficiency without compromising extraction
capabilities, selecting a lowerBis preferable.
Token replacements per step S. The parameter Ssets how
many modifications the Qgen algorithm applies to the base
chunk in a single step. Similar to B, changing Sdoes not show
a clear effect in Table 8. However, increasing Scauses a large
increase in computational time. Specifically, the per-query
generation time jumps from 8.60 s at S=1 to 50.39 s at S=5
(Table 9). This shows that lower values of Sare better for
keeping the execution fast.
Overall, the results for the Qgen parameters M,B, and S
demonstrate that while the algorithm’s extraction capabili-
ties are robust to minor parameter changes, its computational
overhead is sensitive to them. Based on these findings, an
attacker should configure these parameters according to their
constraints. If the objective is to maximize the ER, our evalua-
12

Table 9: Generation times (in seconds) per single query
(Query) and multiple queries from a base chunk (Base) across
different parameter configurations.
Parameter Value Base(s) Query(s)
M200 138.10 23.02
400 137.85 22.97
B6 173.04 28.84
14 412.99 68.83
S1 51.60 8.60
5 302.32 50.39
n4 68.90 17.23
8 135.51 16.94
tions indicate that moderate parameter values (e.g., M=300 ,
B=10,S=3) offer a stable performance.
Adaptive threshold level naffects the number of modi-
fied query variants generated from each extracted chunk. A
higher nwill result in more API calls to the target system and
thus increase the cost. As shown in Table 8, higher nyields a
slight improvement in Query@50% results, modestly reduc-
ing the number of queries needed to reach the 50% extraction.
The overall ER, however, does not exhibit a clear correlation
with n, demonstrating only minor fluctuations of 4% to 6.7%.
However, nhas a definitive impact on generation times. From
Table 9, we observe that although the generation time per
query remains similar, the Base(s) value almost doubles from
68.90 s to 135.51 s as nincreases from 4 to 8. This demon-
strates that the total computation time required to process a
single extracted base chunk increases substantially as more
query variants are requested.
ROUGE-L deduplication threshold τrouge controls how
aggressively the Stolen KB rejects near-paraphrases of
already-stored chunks. A higher τrouge lets more chunks slip
through, which might cause redundant query generation and
waste attacker budget, while a lower threshold might reject
chunks that are actually new. Our ablation results show no
significant difference in performance across the tested τrouge
values of 0.55, 0.65, and 0.75. Query@50% ranges narrowly
from 447 to 453, showing that the extraction pipeline is robust
toτrouge within a standard operational range, allowing an at-
tacker to rely on a moderate default value (e.g., τrouge=0.55 ).
Extraction mechanism.To demonstrate the effectiveness
of our extraction mechanism when combined with the injec-
tion command, we compare it against a hypothetical oracle
baseline. This oracle always returns the chunks that the tar-
get LLM uses to generate responses. This eliminates any
extraction failures and LLM hallucinations. Figure 6 demon-
strates that both settings follow a similar performance, which
proves our extraction mechanism’s and injection command’s
effectiveness. To extract half of the target database, Oracle
Figure 6: Default regex-based extractor versus an oracle that
returns retrieved chunks without parsing error. The two curves
are nearly indistinguishable, which shows that our extraction
mechanism with our injection command successfully steals
chunks from the LLM’s response.
extraction achieves 412 Query@50%, and the regex extractor
achieves 453 Query@50%, which presents only a slight im-
provement due to the absence of extraction errors. We present
the results of the compared attacks combined with our ex-
traction mechanism in Appendix D, showing that it improves
performance in some scenarios.
6 Conclusion
In this work, we introducedSNAPRAG: a novel, black-box
knowledge extraction attack on RAG systems that eliminates
the dependency on an external LLM. Unlike prior methods
that rely on external LLMs for query generation, which natu-
rally require extra resources, our method requires significantly
fewer queries to extract knowledge from RAG systems. By
combining a discrete token mutation strategy with an adap-
tive dissimilarity threshold, a multi-level deduplication filter,
and a ranked dispatch mechanism, our method systematically
generates queries in unexplored semantic regions of the target
knowledge base and thereby ensures the attacker’s budget is
spent effectively. Experiments show thatSNAPRAG consis-
tently outperforms state-of-the-art LLM-dependent attacks
under limited query budgets, achieving higher ER and QE. Un-
der an active input and output moderation defense,SNAPRAG
remains the leading attack on four of five targets and recovers
50.7-96.3% of their knowledge bases.
Although moderation reduces extraction in some settings, it
does not consistently prevent recovery. This finding exposes a
substantial privacy risk in current RAG systems and motivates
defenses designed specifically to prevent such adaptive extrac-
tion attacks. We plan to work on defenses in future work. We
also plan to extendSNAPRAG to graph-based [6, 11, 28] and
multi-modal RAG systems, where different retrieval mecha-
nisms and prompt structures may introduce new challenges
for knowledge extraction.
13

References
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach, and
Amit Bahree et al. Phi-3 technical report: A highly
capable language model locally on your phone, 2024.
arXiv:2404.14219.
[2]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub-
biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Nee-
lakantan, Pranav Shyam, and Sastry et al. Language
models are few-shot learners. InAdvances in Neural
Information Processing Systems, volume 33, pages 1877–
1901. Curran Associates, Inc., 2020.
[3]Bochuan Cao, Changjiang Li, Yuanpu Cao, Yameng Ge,
Ting Wang, and Jinghui Chen. You can’t steal nothing:
Mitigating prompt leakages in llms via system vectors.
Proceedings of the 2025 ACM SIGSAC Conference on
Computer and Communications Security, 2025.
[4]Stav Cohen, Ron Bitton, and Ben Nassi. Unleashing
worms and extracting data: Escalating the outcome of
attacks against rag-based inference in scale and severity
using jailbreaking.arXiv preprint arXiv:2409.08045,
2024.
[5]Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing
Dou. HotFlip: White-box adversarial examples for text
classification. InProceedings of the 56th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 2: Short Papers), pages 31–36, Melbourne, Aus-
tralia, July 2018. Association for Computational Lin-
guistics.
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
Alex Chao, Apurva N. Mody, Steven Truitt, and
Jonathan Larson. From local to global: A graph
rag approach to query-focused summarization.ArXiv,
abs/2404.16130, 2024.
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin
Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng
Wang, and Haofen Wang. Retrieval-augmented gen-
eration for large language models: A survey.ArXiv,
abs/2312.10997, 2023.
[8]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, and Ahmad Al-
Dahle et al. The llama 3 herd of models, 2024. arXiv:
2407.21783.
[9]Kai Greshake, Sahar Abdelnabi, Shailesh Mishra,
Christoph Endres, Thorsten Holz, and Mario Fritz. Not
what you’ve signed up for: Compromising real-world
llm-integrated applications with indirect prompt injec-
tion. InProceedings of the 16th ACM Workshop on Arti-
ficial Intelligence and Security, AISec ’23, page 79–90,New York, NY , USA, 2023. Association for Computing
Machinery.
[10] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Ming-Wei Chang. Realm: Retrieval-augmented
language model pre-training.ArXiv, abs/2002.08909,
2020.
[11] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen
Ling, and Liang Zhao. Grag: Graph retrieval-augmented
generation. InFindings of the Association for Compu-
tational Linguistics: NAACL 2025, pages 4145–4157,
2025.
[12] Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, and
Yinzhi Cao. Pleak: Prompt leaking attacks against large
language model applications.Proceedings of the 2024
on ACM SIGSAC Conference on Computer and Com-
munications Security, 2024.
[13] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi
Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev,
Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama
guard: Llm-based input-output safeguard for human-ai
conversations.arXiv preprint arXiv:2312.06674, 2023.
[14] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave.
Atlas: Few-shot learning with retrieval augmented lan-
guage models.Journal of Machine Learning Research,
24(251):1–43, 2023.
[15] Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao,
Yang Chen, and Min Yang. Feedback-guided extrac-
tion of knowledge base from retrieval-augmented llm
applications.arXiv preprint arXiv:2411.14110, 2024.
[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez,
Hao Zhang, and Ion Stoica. Efficient memory manage-
ment for large language model serving with pagedatten-
tion. InProceedings of the 29th symposium on operating
systems principles, pages 611–626, 2023.
[17] Yaniv Leviathan, Matan Kalman, and Yossi Matias.
Prompt repetition improves non-reasoning llms.arXiv
preprint arXiv:2512.14982, 2025.
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474, 2020.
14

[19] Fan Li, Sijia Xu, Mengting Pan, Chen Chen, Ying Zhang,
Wenjie Zhang, and Xiaoyang Wang. A survey of knowl-
edge extraction attacks and defenses in llm agent ecosys-
tems. 2026.
[20] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. Towards general text
embeddings with multi-stage contrastive learning, 2023.
arXiv:2308.03281.
[21] Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Tianwei
Zhang, Yepang Liu, Haoyu Wang, Yanhong Zheng, and
Yang Liu. Prompt injection attack against llm-integrated
applications.ArXiv, abs/2306.05499, 2023.
[22] Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and
Neil Zhenqiang Gong. Formalizing and benchmarking
prompt injection attacks and defenses. In33rd USENIX
Security Symposium (USENIX Security 24), pages 1831–
1847, Philadelphia, PA, August 2024. USENIX Associ-
ation.
[23] Christian Di Maio, Cristian Cosci, Marco Maggini,
Valentina Poggioni, and Stefano Melacci. Pirates of
the rag: Adaptively attacking llms to leak knowledge
bases. InEuropean Conference on Artificial Intelligence,
2025.
[24] Luke Merrick, Danmei Xu, Gaurav Nuti, and Daniel
Campos. Arctic-embed: Scalable, efficient, and accurate
text embedding models.ArXiv, abs/2405.05374, 2024.
[25] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, and Sam Altman
et al. Gpt-4 technical report, 2024. arXiv:2303.08774 .
[26] Fábio Perez and Ian Ribeiro. Ignore previous prompt:
Attack techniques for language models.arXiv preprint
arXiv:2211.09527, 2022.
[27] Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade,
and Himabindu Lakkaraju. Follow my instruction
and spill the beans: Scalable data extraction from
retrieval-augmented generation systems.arXiv preprint
arXiv:2402.17840, 2024.
[28] Priyanka Sen, Sandeep Mavadia, and Amir Saffari.
Knowledge graph-augmented language models for com-
plex question answering. InProceedings of the 1st Work-
shop on Natural Language Reasoning and Structured
Explanations (NLRSE), pages 1–8, Toronto, Canada,
June 2023. Association for Computational Linguistics.
[29] Zeyang Sha and Yang Zhang. Prompt stealing attacks
against large language models.ArXiv, abs/2402.12959,
2024.[30] Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. Beir: A heteroge-
nous benchmark for zero-shot evaluation of informa-
tion retrieval models.arXiv preprint arXiv:2104.08663,
2021.
[31] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao,
Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu
Wei. Text embeddings by weakly-supervised contrastive
pre-training, 2024.arXiv:2212.03533.
[32] Yuhao Wang, Wenjie Qu, Shengfang Zhai, Yanze Jiang,
Liu Zichen, Yue Liu, Yinpeng Dong, and Jiaheng Zhang.
Silent leaks: Implicit knowledge extraction attack on
rag systems. InInternational Conference on Learn-
ing Representations, volume 2026, pages 24150–24191,
2026.
[33] Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed
resources for general chinese embeddings. InProceed-
ings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval,
SIGIR ’24, page 641–649, New York, NY , USA, 2024.
Association for Computing Machinery.
[34] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, and Chengpeng Li et al. Qwen2
technical report.arXiv preprint arXiv:2407.10671,
2024.
[35] Yi Yang, Wen-tau Yih, and Christopher Meek. Wikiqa:
A challenge dataset for open-domain question answer-
ing. InProceedings of the 2015 conference on empirical
methods in natural language processing, pages 2013–
2018, 2015.
[36] Mengyu Yao, Ziqi Zhang, Ning Luo, Shaofei Li, Yifeng
Cai, Xiangqun Chen, Yao Guo, and Ding Li. Con-
nect the dots: Knowledge graph–guided crawler attack
on retrieval-augmented generation systems. In35th
USENIX Security Symposium (USENIX Security 26),
2026.
[37] Jingwei Yi, Yueqi Xie, Bin Zhu, Keegan Hines, Emre
Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu.
Benchmarking and defending against indirect prompt
injection attacks on large language models.Proceedings
of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V .1, 2023.
[38] Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang Wang,
Dawei Yin, et al. The good and the bad: Exploring pri-
vacy issues in retrieval-augmented generation (rag). In
Findings of the Association for Computational Linguis-
tics: ACL 2024, pages 4505–4524, 2024.
15

[39] Yiming Zhang, Nicholas Carlini, and Daphne Ippolito.
Effective prompt extraction from language models.
arXiv preprint arXiv:2307.06865, 2023.
A Ethical Considerations
In this paper, we present a practical knowledge extraction
attack on RAG systems. AlthoughSNAPRAG is an attack
method, we believe that publicly disclosing these vulnera-
bilities is crucial to advancing AI safety. By demonstrating
how efficiently data can be extracted, we aim to highlight the
practical security vulnerabilities of current RAG systems, and
we hope that our research will encourage the development of
stronger defenses in future RAG pipelines.
To ensure that our research was conducted ethically, all
experiments were run in safe, locally hosted environments.
We did not target commercial RAG systems, live APIs, or
private user data. Additionally, the databases we attacked were
built entirely from public datasets, generating a simulation
environment. This allowed us to demonstrate the effectiveness
of our attack without putting real sensitive information at risk.
B More Details Regarding Experiment Setup
B.1 Target RAG Agents
To ensure a fair comparison with existing methods, we create
an experimental setup similar to PirateRAG [23]. We define
various target RAG agents, each of which is a combination of
various LLMs, embedding models, the number of retrieved
documents (top-k), and private databases with different do-
mains. Table 2 summarizes our configurations. All agents
follow the standard retrieve-then-generate pipeline: given a
user query, the system retrieves the top- kmost relevant docu-
ments from its knowledge base using the specified embedder,
injects them into a prompt template, and generates a response
with the LLM. We reproduce two agents of PirateRAG, two
agents from the experimental setup of RAGCrawler, and add
one additional agent with a new dataset to evaluate gener-
alization. All agents are served via vLLM [16] for efficient
inference, exposing a single HTTP endpoint that accepts a
query and returns the generated response.
Agent A (BioASQ)uses 1,000 biomedical passages from
the Mini-BioASQ,3embedded with Alibaba-NLP/gte-large-
en-v1.5 [20], retrieving k=3 documents per query. The LLM
is Meta Llama-3.2-3B-Instruct [8].
Agent B (Wikipedia)uses 1,000 passages from the Mini-
Wikipedia,4embedded with intfloat/e5-large-v2 [31], with k=
5 retrievals. The LLM is Microsoft Phi-3.5-mini-instruct [1].
3https://huggingface.co/datasets/rag-datasets/
rag-mini-bioasq
4https://huggingface.co/datasets/rag-datasets/
rag-mini-wikipediaAgent C (Neural Bridge)is a new agent we introduce
to evaluate generalization beyond the datasets used in prior
work. Following the creation mechanism of other agents [23],
we construct a knowledge base of 1,000 passages sampled
from the Neural Bridge RAG dataset,5a diverse collection of
question–context–answer triples spanning multiple domains.
The embedding model is BAAI/bge-large-en-v1.5 [33] with
k=5 retrievals, and the LLM is Meta Llama-3.2-3B-Instruct.
Agents D and E (TREC-COVID and SciDocs)use 1,000
passages from the TREC-COVID and SciDocs datasets,
which are from the BEIR [30] benchmark. Following the
experimental setup of [36], these settings use the BGE fam-
ily embedding model, Meta Llama-3.1-8B-Instruct [8] LLM,
withk=10.
B.2 Attacker Configuration
To ensure a fair comparison, we used the same com-
mon attacker configurations as those in [23], including
the attacker-side embedding model and deduplication pa-
rameters. The attacker uses Snowflake Arctic Embed
Large [24] ( snowflake-arctic-embed-l ) as its local em-
bedding model for all operations: Stolen KB indexing, query
history tracking, and computations in Qgen . This model is
differentfrom every target agent’s embedder, reinforcing the
black-box nature of the attack. We set the semantic deduplica-
tion threshold to τdedup=0.95 . The ROUGE-L deduplication
threshold is set to τrouge=0.55 , checked against the kr=5
nearest neighbours. The query-level deduplication threshold
isτqdedup =0.95.
Table 3 provides the default values of the hyperparameters
for our Qgen algorithm. We select these values to balance
the attacker’s local computational budget with the necessary
precision of the embedding space exploration. Parameters M
andBdetermine the granularity of the search, controlling how
finely the algorithm navigates toward the novelty objective.
The number of adaptive threshold levels nis determined by
the attacker’s financial constraints regarding the target API.
Since nsets a hard upper bound on the number of novel
queries generated per extracted chunk, it allows the attacker
to explicitly control their query spending in relation to the
amount of stolen data.
B.3 Compared Approaches
For experimental comparisons, we use the open-source imple-
mentations provided by [23] for RAGThief and PirateRAG.
For RAGThief [15], we use the provided implementation di-
rectly. For PirateRAG [23], we restrict the algorithm to use
only the first injection command from its default injection
pool. This adjustment is necessary to ensure a fair compar-
ison in budget-limited scenarios; evaluating a pool of mul-
5https://huggingface.co/datasets/neural-bridge/
rag-dataset-12000
16

Table 10: Llama Guard 3 hazard classes assigned toSNAPRAG queries. Counts include unsafe input and output decisions. Guard
execution errors are excluded from the class rows and presented separately.
Class Hazard category Agent A Agent B Agent C Agent D Agent E
S1 Violent Crimes 1 16 8 1 1
S2 Non-Violent Crimes 1 1 11 0 0
S3 Sex-Related Crimes 0 0 0 0 0
S4 Child Sexual Exploitation 7 16 13 0 0
S5 Defamation 3 120 30 0 0
S6 Specialized Advice 248 4 26 61 1
S7 Privacy 3 23 49 0 1
S8 Intellectual Property 5 142 41 1 5
S9 Indiscriminate Weapons 16 6 1 3 1
S10 Hate 6 405 24 1 2
S11 Suicide & Self-Harm 2 4 1 0 0
S12 Sexual Content 1 5 2 0 0
S13 Elections 3 44 9 0 1
S14 Code Interpreter Abuse 15 30 1 0 0
– Unsafe classifications 311 816 216 67 12
– Guard errors 22 0 0 0 0
– Total blocked calls 333 816 216 67 12
tiple injection commands for a single query inherently mul-
tiplies the number of API calls made to the target, and do-
ing so severely inflates the query budget for PirateRAG. For
RAGCrawler [36], we design Agents D and E to replicate its
two experimental setups. We use the official implementation
of [36], with Qwen2.5-7B-Instruct [34] as the attacker side
LLM, and discuss the results separately from the other works
in Section 5.4 because of the stated reasons.
C Analysis of Guard Rejections
To investigate the reasons behind guard rejections and un-
derstand agent-to-agent different in guard rejection rates, we
study the moderation metadata. Llama Guard 3 assigns un-
safe inputs and outputs to 14 hazard classes [8]. Table 10
presents these classifications across Agents A-E. Each unsafe
decision in our runs contains exactly one class, so the class
counts sum to the number of unsafe classifications. These
classes are model-generated moderation decisions rather than
human-validated labels of the dispatched query or retrieved
content. The 22 guard execution errors on Agent A are re-
ported separately.
The biomedical configurations show concentrated class dis-
tributions. On Agent A, S6 (Specialized Advice) accounts for
248 of 311 unsafe classifications (79.7%). On Agent D, it ac-
counts for 61 of 67 (91.0%). This concentration is consistent
with the medical and scientific passages retrieved by these
agents being evaluated as part of the full prompt. However,
Agent A and Agent D have different total block rates (16.65%
and 6.70%, respectively). A shared broad domain is therefore
not sufficient to predict rejection; the exact combination ofthe retrieved passages, query, injection command, and, for
output moderation, generated response may also contribute.
Agent B has the highest rejection rate and a high-volume
distribution across several classes. S10 (Hate) accounts for
405 of its 816 unsafe classifications (49.6%), followed by S8
(Intellectual Property) with 142 (17.4%) and S5 (Defamation)
with 120 (14.7%). Furthermore, 124 of its 132 output blocks
are assigned S10. One plausible explanation is the context
of the Wikipedia-based knowledge base: retrieved passages
cover people, historical conflicts, protected groups, elections,
and other topics that interact with several guard classes. The
fixed injection command also requests verbatim reproduction,
which contributes to S8 decisions. However, the same com-
mand is used for every agent, so it cannot by itself explain
Agent B’s higher rejection rate. These observations support
the impact of retrieved context in the final guard decision.
Agent C also reflects its mixed-domain configuration: its
most frequent classes are S7 (Privacy, 49), S8 (Intellectual
Property, 41), S5 (Defamation, 30), S6 (Specialized Advice,
26), and S10 (Hate, 24), without a single class dominating
as strongly as on Agents A, B, or D. Agent E produces only
12 unsafe classifications, of which five are S8. This count
is too small to support a strong conclusion about its class
distribution.
Overall, the results suggest that the guard’s rejection pat-
terns are knowledge base-dependent, with different safety cat-
egories occurring more frequently across different domains,
influenced by the nature of the underlying knowledge base.
17

Table 11: Number of queries required to extract 50% of the
target knowledge base (Query@50%). The values in paren-
theses indicate the relative query cost multiplier compared to
SNAPRAG. Best results are in bold. Lower is better.
Attack Agent A Agent B Agent C
RAGThief+SNAP3409 (3.42×) † 2624 (5.79×)
PirateRAG+SNAP1291 (1.29×)855(0.91×) ‡
SNAPRAG (Ours) 997(1.00×) 942 (1.00×)453(1.00×)
†: The attack automatically terminated before reaching 50% extraction.
‡: We halted the attack after 10,000 queries as it failed to reach 50% extraction.
D Evaluation of Query Generation Strategies
Evaluation setup.To isolate the performance of the under-
lying query generation algorithms from the effectiveness of
prompt injection payloads, we introduce two modified attack
variants: PirateRAG+SNAPand RAGThief+SNAP. In these
variants, we replace their default injection commands and ex-
tractors with our own, indicated as "+SNAP". This decoupling
ensures that the empirical comparison is entirely independent
of the design of the injection commands, enabling a direct
evaluation of each method’s query generation strategy.
Results and discussion.Table 11 demonstrates that al-
though incorporating our extraction mechanism slightly de-
grades performance in some cases, it substantially improves
PirateRAG’s results on Agents A and B. Specifically, Pirat-
eRAG+SNAPreaches 50% extraction with only 1291 and 855
queries to the target RAG, compared to the original Pirat-
eRAG’s 1379 and 1671 queries on Agents A and B, respec-
tively. We observe that whileSNAPRAG still outperforms the
+SNAPversions of its competitors on Agents A and C, Pirat-
eRAG+SNAPperforms better on Agent B, requiring 0.91 ×
fewer queries thanSNAPRAG. On the other hand, in Table
12, we observe that under a 2000 query budget,SNAPRAG
outperforms all competitors with 69.7%, 68.5%, and 79.3%
ER, compared to the closest results of 57.4%, 68.1%, and
49.2% on Agents A, B, and C, respectively. In this table, we
can also see that the results for RAGThief+SNAPsignificantly
increase from 16.5% to 44.6% compared to RAGThief, and
the attack does not terminate early on Agent A.
These results underscore the substantial standalone value
of our query generation and extraction mechanisms. The fact
that PirateRAG’s query cost is almost cut in half by adopt-
ing our prompt injection payload shows that our extraction
framework is highly robust and transferable. This proves
thatSNAPRAG offers two distinct, valuable contributions:
a budget-efficient query generator and a robust, plug-and-play
extraction pipeline that can significantly improve the success
rates of existing LLM-dependent attacks.Table 12: Extraction Rate and Query Efficiency results under
a 2000-query budget for Agents A, B, and C. Best results are
in bold. Higher ER and lower QE are better.
Agent A Agent B Agent C
AttackER QE ER QE ER QE
RAGThief+SNAP44.6% 4.48 17.4% 11.49 49.2% 4.07
PirateRAG+SNAP57.4% 3.48 68.1% 2.94 26.9% 7.43
SNAPRAG (Ours) 69.7% 2.87 68.5% 2.92 79.3% 2.52
E Computational Cost and API Pricing
We investigate whether the query-efficiency advantage of
SNAPRAG translates into lower API costs, and what attacker-
side computation is required to obtain this advantage. Al-
thoughSNAPRAG avoids using an attacker-side LLM, its
query search requires repeated token mutations and embed-
ding evaluations. We therefore compare victim API usage,
attacker-side LLM usage, and mean elapsed times to assess
the trade-off between API cost and local computation.
Evaluation setup.We evaluateSNAPRAG, RAGThief and
PirateRAG across all five datasets, accounting for the work
performed until each attack first reaches 50% extraction. Ta-
ble 13 estimates the API charges for different providers, while
Table 14 reports the measured workloads and runtimes. Mean
elapsed times include attacker preparation, local processing,
and waiting for model responses; they exclude prior server and
index setup and offline evaluation. We use oracle to denote
the attacker-side LLM employed by the competing methods.
All cost experiments use two NVIDIA A40 GPUs, with one
assigned to the victim and the other to attacker-side computa-
tion.
API cost estimates.To determine how the measured work-
loads affect API charges, we use the recorded input and out-
put token counts, including repeated chunks and unsuccessful
queries. Successful near-verbatim extraction requires the vic-
tim to receive retrieved chunks in its context and reproduce
chunk text in its output. Both input and output usage there-
fore contribute to the attack cost. Let Iv,OvandIa,Oadenote
the victim and oracle token counts, respectively. Given their
input/output prices pv,i,pv,o,pa,i,pa,oper million tokens, the
combined LLM charge is
CLLM=pv,iIv+p v,oOv+p a,iIa+p a,oOa
106.(5)
Table 13 applies different per-token prices to the same
recorded workloads. This estimates the effect of pricing while
holding the observed query sequences and token counts fixed.
It does not establish extraction performance on the listed
models, which may differ in tokenization, responses, and la-
tency. We report victim and oracle charges separately because
the attacker can choose an oracle independently of the vic-
tim, including a locally hosted model with no oracle API fee.
18

For the API prices in Table 13, we use OpenAI’s standard un-
cached short-context prices and Together AI’s listed serverless
prices.67
Even when oracle API fees are excluded,SNAPRAG has
the lowest victim charge among the reported entries on
BioASQ, Neural Bridge, and SciDocs under every listed tariff.
It is also the lowest or tied on Wikipedia and TREC-COVID
when input and output tokens have equal prices. We observe
some exceptions on Wikipedia and TREC-COVID datasets
when output tokens are more expensive. At $2/$10 per mil-
lion input/output tokens,SNAPRAG costs $11.56 versus Pirat-
eRAG’s $10.98 on Wikipedia, and $8.43 versus RAGThief’s
$7.57 on TREC-COVID. In these cases,SNAPRAG sends
fewer queries but produces more output tokens, leaving its
victim API cost slightly higher. This explains why query effi-
ciency alone does not determine the API bill.
If both the victim and oracle use paid APIs at the same
listed tariff,SNAPRAG has the lowest combined charge
among the reported entries on every dataset. However, its
API cost advantage does not depend solely on assuming a
paid oracle. Victim-only charges are generally lower and re-
main comparable in the two exceptions above.
Computation and runtime.As shown in Table 14,
SNAPRAG reaches 50% extraction with fewer victim calls
than each competitor that reaches this threshold on the same
dataset. On SciDocs, for example, it requires 214 calls, com-
pared to 466 for RAGThief and 1,614 for PirateRAG. The
competing methods also invoke an oracle to generate their
queries, whereasSNAPRAG has no oracle calls. These results
show thatSNAPRAG reduces interactions with the victim
while avoiding the additional LLM generation workload.
We report mean elapsed time per victim call and per gener-
ated candidate to account for how the methods prepare their
queries. Both averages divide the full measured elapsed time
by the corresponding count, including all preparation and
model responses.SNAPRAG has a higher mean time per vic-
tim call on all five datasets. However, it prepares queries in
bulk, including candidates that have not yet been submitted.
For example, on SciDocs, RAGThief generates 466 candi-
dates for 466 victim calls, whileSNAPRAG generates 2,426
candidates for 214 calls. If these queued candidates are used
later, their existing preparation cost is spread over more victim
calls. We do not subtract this cost from the reported victim
call time values.
When averaged over generated candidates,SNAPRAG has
the lowest reported mean elapsed time on Wikipedia, Neural
Bridge, TREC-COVID, and SciDocs. On SciDocs,SNAPRAG
averages 33.30 seconds per candidate, compared to 72.21 for
RAGThief and 64.13 for PirateRAG. BioASQ is the excep-
tion, whereSNAPRAG averages 40.21 seconds compared to
PirateRAG’s 30.68 seconds.
6https://www.together.ai/pricing, October 12, 2026;
7https://developers.openai.com/api/docs/pricing , October 12,
2026;
19

Dataset MethodLlama 3 8B
Instruct Lite
0.14 / 0.14Llama 3.3
70B
1.04 / 1.04GPT-6
Luna
0.10 / 0.50GPT-6
Sol
2.00 / 10.00GPT-6
Astra
10.00 / 50.00
Victim Oracle Victim Oracle Victim Oracle Victim Oracle Victim Oracle
BioASQRAGThief‡N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
PirateRAG 0.63 0.80 4.71 5.93 1.12 1.63 22.48 32.59 112.41 162.95
SNAPRAG 0.32 0.00 2.40 0.00 0.60 0.00 12.07 0.00 60.34 0.00
WikipediaRAGThief 0.30 0.55 2.20 4.05 0.64 1.72 12.73 34.43 63.64 172.14
PirateRAG 0.27 1.55 2.01 11.48 0.55 3.60 10.98 72.06 54.91 360.29
SNAPRAG 0.26 0.00 1.96 0.00 0.58 0.00 11.56 0.00 57.79 0.00
Neural
BridgeRAGThief†0.52 0.56 3.88 4.16 1.17 1.71 23.32 34.14 116.58 170.72
PirateRAG 0.36 1.58 2.68 11.73 0.81 3.95 16.24 79.06 81.20 395.29
SNAPRAG 0.18 0.00 1.30 0.00 0.34 0.00 6.74 0.00 33.72 0.00
TREC-COVIDRAGThief 0.21 0.13 1.59 1.00 0.38 0.41 7.57 8.28 37.85 41.38
PirateRAG 0.76 0.90 5.67 6.66 1.34 1.82 26.73 36.37 133.65 181.83
SNAPRAG 0.21 0.00 1.55 0.00 0.42 0.00 8.43 0.00 42.17 0.00
SciDocsRAGThief 0.32 0.21 2.38 1.56 0.57 0.64 11.36 12.74 56.79 63.72
PirateRAG 1.13 1.02 8.40 7.57 2.02 2.27 40.42 45.39 202.12 226.97
SNAPRAG 0.17 0.00 1.24 0.00 0.33 0.00 6.69 0.00 33.45 0.00
†RAGThief stopped at 49.3% extraction on Neural Bridge; the values include its complete run of 1,135 victim calls.
‡N/A: RAGThief did not reach 50% extraction on BioASQ (maximum 27.9%).
Table 13: Estimated API charges (USD) at the reported extraction endpoints. Input/output rates below each tariff name are in
USD per million tokens. Victim and oracle charges are shown separately and can be priced independently.
Dataset MethodVictim
callsOracle
generationQuery
candidatesMean elapsed
time (s)Victim tokens
(M)Oracle tokens
(M)
Batches Prompts Call Candidate Input Output Input Output
BioASQRAGThief‡N/A N/A N/A N/A N/A N/A N/A N/A N/A N/A
PirateRAG 1,307 2,107 3,134 1,307 30.68 30.68 2.854 1.678 3.051 2.649
SNAPRAG 910 0 0 2,407 106.35 40.21 1.379 0.931 0.000 0.000
WikipediaRAGThief 1,144 1,144 1,144 1,144 35.57 35.57 1.057 1.061 0.570 3.329
PirateRAG 1,610 2,487 5,813 1,610 24.43 24.43 1.049 0.888 4.788 6.248
SNAPRAG 953 0 0 3,827 94.58 23.55 0.905 0.975 0.000 0.000
Neural
BridgeRAGThief†1,135 1,135 1,135 1,135 43.96 43.96 1.746 1.982 0.728 3.269
PirateRAG 865 1,443 4,121 865 44.99 44.99 1.193 1.385 4.212 7.063
SNAPRAG 444 0 0 2,782 190.47 30.40 0.724 0.530 0.000 0.000
TREC-COVIDRAGThief 260 260 260 260 86.99 86.99 0.962 0.565 0.169 0.794
PirateRAG 941 1,216 3,604 941 74.50 74.50 3.469 1.979 3.459 2.945
SNAPRAG 209 0 0 1,849 439.06 49.63 0.814 0.681 0.000 0.000
SciDocsRAGThief 466 466 466 466 72.21 72.21 1.438 0.848 0.282 1.218
PirateRAG 1,614 1,866 3,959 1,614 64.13 64.13 5.040 3.034 3.430 3.853
SNAPRAG 214 0 0 2,426 377.47 33.30 0.657 0.538 0.000 0.000
†RAGThief stopped at 49.3% extraction on Neural Bridge; the values include its complete run of 1,135 victim calls.
‡N/A: RAGThief did not reach 50% extraction on BioASQ (maximum 27.9%).
Table 14: Model usage, mean elapsed times, and token counts for extracting 50% of each knowledge base, subject to the noted
exceptions. We report oracle generation batches and individual prompts separately, since one batch can process multiple prompts.
Mean times divide the full elapsed time by victim calls or generated query candidates, respectively. M denotes millions.
20