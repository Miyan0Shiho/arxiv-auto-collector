# TEM: Temporal State Tracking Memory for Large Language Model Agents

**Authors**: Pengfeng Song, Yunkai Zhuang, Liye Shi

**Published**: 2026-10-15 13:54:31

**PDF URL**: [https://arxiv.org/pdf/2610.21900v1](https://arxiv.org/pdf/2610.21900v1)

## Abstract
An agent's memory must remain useful as information changes across conversations. A new fact may replace an earlier value without making that value irrelevant to a later question. We present Temporal State Tracking Memory (TEM), a training-free framework for tracking these changes. TEM stores facts as versioned records linked to their sources. It retrieves relevant versions, places them in source order and labels historical records before passing the evidence to the answer model. Event memory and original dialogue turns provide details missing from structured facts. We evaluated TEM on 420 questions drawn from 600 sessions involving 60 personas. Across four open-weight models, key-entity hit rates reached 75.71-97.62% overall and 96.67-100.00% on the historical-residence subset. Adding historical labels to otherwise identical, source-ordered facts increased overall hit rates by 17.14-55.95 percentage points. The evidence was compact: 94-118 tokens per answer, or 48.75-56.73% less than session-aware retrieval-augmented generation (RAG). A further evaluation on 1,540 LoCoMo questions tested hybrid memory under a fixed evidence budget. TEM exceeded session-aware RAG with both backbones; conservative score differences were 7.92 and 10.58 percentage points. The results support preserving fact versions and exposing their temporal status to the generator when answering questions about repeatedly updated information.

## Full Text


<!-- PDF content starts -->

TEM: Temporal State Tracking Memory for Large
Language Model Agents
Pengfeng Song1*, Yunkai Zhuang1and Liye Shi1
1School of Information and Intelligent Engineering,Zhejiang Wanli
University,Ningbo,315100,China.
*Corresponding author(s). E-mail(s): sungkites@163.com;
Abstract
An agent’s memory must remain useful as information changes across conversa-
tions. A new fact may replace an earlier value without making that value irrelevant
to a later question. We present Temporal State Tracking Memory (TEM), a
training-free framework for tracking these changes. TEM stores facts as versioned
records linked to their sources. It retrieves relevant versions, places them in source
order and labels historical records before passing the evidence to the answer model.
Event memory and original dialogue turns provide details missing from structured
facts. We evaluated TEM on 420 questions drawn from 600 sessions involving
60 personas. Across four open-weight models, key-entity hit rates reached 75.71–
97.62% overall and 96.67–100.00% on the historical-residence subset. Adding
historical labels to otherwise identical, source-ordered facts increased overall hit
rates by 17.14–55.95 percentage points. The evidence was compact: 94–118 tokens
per answer, or 48.75–56.73% less than session-aware retrieval-augmented gener-
ation (RAG). A further evaluation on 1,540 LoCoMo questions tested hybrid
memory under a fixed evidence budget. TEM exceeded session-aware RAG with
both backbones; conservative score differences were 7.92 and 10.58 percentage
points. The results support preserving fact versions and exposing their tempo-
ral status to the generator when answering questions about repeatedly updated
information.
Keywords:large language model agents; long-term memory; temporal state tracking;
fact versions; retrieval-augmented generation
1
arXiv:2610.21900v1  [cs.CL]  15 Oct 2026

1 Introduction
A long-term agent accumulates records that may no longer describe its user. Residence,
occupation, employer and preferences can change several times across sessions [ 1–3].
Earlier records still matter when a question refers to the past. The agent therefore
needs to know which version applies to the question, including whether the latest
record is relevant at all.
Suppose a user moves from Shanghai to Chengdu and then to Suzhou. Suzhou
answers “Where does the user live now?” The question “Where did the user live before
moving to Chengdu?” requires Shanghai. A memory that keeps only the latest address
cannot answer both. Keeping all three addresses makes the historical answer available,
but the model still has to identify the right one.
Persistent memory, action feedback and experience-based reflection have extended
agents’ use of past interactions [ 4–6]. Work on hierarchical storage, memory updates,
associative retrieval and temporal graphs has also improved access to this information
[7–14]. Repeated updates pose a particular problem: several retrieved values may be
equally relevant to a question yet refer to different states. We examine how the memory
can convey these version relationships to the answering model.
TEM records cross-session changes as fact versions without retraining the backbone.
New facts are appended, and the status of earlier records is updated from their source
order. At query time, relevance scores select the evidence. TEM then orders the selected
facts by source and labels historical versions. This separates evidence selection from
the presentation of temporal information. Event memory and original turns supply
details that the fact records do not capture.
Our contribution is a history-preserving update rule that assigns state over complete
fact-version groups. The evidence construction procedure carries this status into
retrieval output, so an older record stays historical when the latest version is not
retrieved. We evaluate both components across multiple backbones and examine hybrid
memory on public multi-session questions.
2 Related Work
2.1 Long-Term Agent Memory and Structured Organisation
External memory lets an agent read and write information outside its model parameters.
Hierarchical management in MemGPT and the historical storage mechanisms studied in
MemoryBank and long-term memory augmentation provide ways to retain information
beyond the immediate context [ 7–9]. A-MEM, Mem0 and MemoryOS extend this
work through memory associations, updates and hierarchical organisation [ 10–12].
Temporal information enters the memory itself in Zep’s knowledge graph [ 13], making
it particularly relevant to state tracking. TEM examines how version status is carried
from stored records into answer evidence. Memory processing cost is another concern;
LightMem addresses it through staged operations [14].
Graph-based retrieval connects information spread across records. Associative
retrieval is central to HippoRAG and its successor, while GraphRAG uses graph
organisation for query-focused summarisation [ 15–17]. Beyond these retrieval structures,
2

MemoRAG uses global memory to guide evidence access, and MemOS considers
knowledge access across the memory lifecycle [18, 19].
KG-RAG combines knowledge-graph retrieval and question decomposition for multi-
hop question answering without additional training [ 20]. CTRN addresses complex time
constraints through implicit temporal and relational representations [ 21]. MGIT models
temporal dependencies at multiple granularities [ 22]. These graph-based question-
answering methods are relevant to TEM’s use of structured evidence. Our focus is
the management of fact versions as conversations update a state, including how the
resulting status information reaches the generator.
2.2 Long-Term Memory Evaluation and Retrieval
Augmentation
LongMemEval and LoCoMo evaluate cross-session information use, fact updates and
temporal reasoning [ 23,24]. LongBench covers a broader set of long-context tasks [ 25].
Lost in the Middle shows that the position of relevant information can affect model
performance [ 26]. For state tracking, finding a relevant passage does not settle which
version answers the question. We study this choice in controlled update tasks and
evaluate hybrid retrieval on public multi-session questions.
RAG and dense retrieval provide the basis for accessing external evidence [ 27,28].
Self-RAG evaluates evidence, RAPTOR organises it hierarchically, and interleaved
retrieval supports chain-of-thought reasoning [ 29–31]. LLMLingua, LongLLMLingua
and RECOMP reduce the input passed to the generator [ 32–34]. TEM retains version
order and state attributes alongside retrieved facts, allowing the generator to interpret
the selected evidence in its temporal context.
3 Temporal State Tracking Memory
TEM stores each fact with its provenance and places successive values in a version
group. Source order determines the status of each version. Retrieval then selects the
facts to include in the answer evidence. Figure 1 gives the overall framework; Fig. 2
and Algorithm 1 detail the update and retrieval procedure.
3.1 Fact Versions with Provenance
A fact record contains the subject, relation, object, source session and source index.
Records with the same subject and relation belong to one version group:
f= (s, r, o, σ, τ), g(f) = (norm(s), r) (1)
The source session identifies the original record. Its index determines the position
of that record in the version sequence. We group records by subject and relation,
normalising subject names to lowercase. Excluding the object from the key lets different
values of a relation remain in the same group.
An update changes a record’s status but leaves its value and provenance intact.
The rule assumes single-valued updates arriving in session order. Source indices record
3

                                   
               
  
  
                   
          
              
                                 
                         
      
      
                           
                                         
                       
                              
                  
                  
                    
                    
                    
                 
                
      
          
          
                                      Fig. 1TEM framework. Fact versions are linked to source sessions. Retrieved facts appear in source
order, with historical labels on superseded versions. Dashed connections supply event and original-text
evidence.
when information entered the memory; they do not establish when the underlying
event took effect.
3.2 History-Preserving Incremental Updates
TEM appends facts from each new session to the existing version set without overwriting
earlier values (Fig. 2a):
Ft=Ft−1∪AttachSource(∆F t, σt) (2)
A frozen model extracts facts from the new session. TEM checks their format and
consistency with the source text before attaching the session provenance.
TEM determines status from the complete version group before selecting evidence.
A version is historical if its source index is smaller than the maximum index in that
subject–relation group:
τ∗(g) = max{τ(f) :f∈F g},
h(f) =1[τ(f)< τ∗(g(f))].(3)
The historical attribute is one for a superseded version and zero when no later
source exists. Both the value and its provenance remain in memory. A missed retrieval
of the latest version therefore cannot make an older record appear current.
The label identifies a record as historical; its source index places it among the
earlier versions. All records at the maximum source index are treated as current. This
rule leaves conflicting values within the same session unresolved.
3.3 Relevance Retrieval and Temporal Evidence
Retrieval selects facts by relevance without penalising older versions for their age. TEM
then places the selected facts in source order (Fig. 2b). The query, source order and
version status are available to the generator when it chooses an answer.
4

Candidate scores combine similarity from the shared vector retriever with a weight
for the relation type:
u(f, q) = cos(e(f), e(q)) 
0.4 + 0.6w(f)
(4)
The relation weights are 1.0 for residence and employer, 0.9 for occupation, 0.8 for
possession, and 0.6 for likes and dislikes. We apply this scoring rule to all versions and
retrieve at most 15, taking those with the highest scores.
TEM sorts the selected records by ascending source index, keeping the fact content
and source session. Superseded versions receive a [historical] label. The latest version
has no label. Evidence construction and answering follow:
Bq= Render(Sort τ(TopK(F, q)), σ, h),
a= LLM(q, B q).(5)
The retrieval budget counts fact records. Sorting and labelling leave their values
unchanged. TEM does not use the question to exclude current or historical versions in
advance; the generator chooses among the retrieved versions using the query and the
supplied temporal information.
3.4 Update and Evidence Construction Algorithm
Algorithm 1 specifies version updates and retrieval, using frozen models for fact
extraction and answer generation.
Algorithm 1Fact-version updates and temporal evidence construction
Input:stored versionsF; new sessionDand sourceσ; optional queryq.
Parameters:candidate countk; relation weightsw.
Output:updatedF; evidenceBand answeraif queried.
1: ∆F←ExtractAndValidate(D, F)
2:F←F∪AttachSource(∆F, σ)
3:foreach subject–relation groupg∈Groups(F)do
4:τ∗(g)←max{τ(f) :f∈F g}
5:foreach versionf∈F gdo
6:h(f)←1[τ(f)< τ∗(g)]
7:end for
8:end for
9:ifno querythen returnF
10:C←TopK(F, q, k), scored by Eq. (4)
11:C←Sort τ(C), ascending source index
12:B←∅
13:foreach candidatef∈Cdo
14:b←RenderFactAndSource(f)
15:ifh(f) = 1thenappend[historical]
16:B←Concat(B, b)
17:end for
18:a←FrozenLLM(q, B)
19:returnF, B, a
AttachSource links each fact to its source session and index.
5

a  Version update b  Evidence construction
Same subject and relation
v1
s1v2
s4
Historical Current
Append v3
v1
s1
Historicalv2
s4
Historicalv3
s7
CurrentQuery Top-k by relevance
Source-ordered evidence
v1 | s1  [historical]
v2 | s4  [historical]
v3 | s7
Frozen LLMFig. 2Fact-version updates and evidence construction. a, Appending a new version updates the
status of retained records. b, Retrieval selects relevant facts, which are then ordered by source and
labelled if historical. Symbols v and s denote a fact version and its source session.
3.5 Event Memory and Original-Text Evidence
Some information does not fit a structured fact value, including event context and
conditions on behaviour. TEM stores event summaries, free-text memories and original
turns for this purpose. Source-session links connect these records to fact versions.
Hybrid retrieval draws on both sources of evidence.
We use the fact-memory configuration TEM-Status for state-update experiments and
TEM-Hybrid-State for public question answering. The hybrid configuration combines
the evidence sources under a common budget. Its unmarked control uses the same
candidates in the same order (Sections 4.1 and 4.6).
4 Experiments and Results
4.1 Experimental Setup
The state-update benchmark contains 60 personas, 600 sessions and 420 questions
involving residence, occupation and drinks. There are 180 current-state questions, 180
other historical questions and 60 questions about residence immediately before the
latest move. For a historical question, the reference answer is the most recent applicable
fact before the target update. Running six memory configurations with four backbones
produced 10,080 answers.
We compared TEM with naive RAG, session-aware RAG and full context. Session-
aware RAG keeps retrieved source turns in session order. The fact-memory comparison
uses three configurations. Overwrite keeps only the latest version. Unmarked keeps all
versions in source order, and TEM-Status adds historical labels to the same content
without changing that order.
Experiments used an A100 40 GB GPU. Extraction and answering within a con-
figuration used the same frozen backbone. We retrieved at most 15 candidates with
all-mpnet-base-v2 and set temperature to zero and the random seed to 42. Llama and
Qwen ran in BF16 with a 512-token answer limit; Qwen thinking was disabled. For
6

gpt-oss, we used MXFP4, low reasoning effort and a 4,096-token answer limit. All
backbones had an 8,192-token extraction output limit.
We measured state-update performance by key-entity hit rate, counting an answer
as a hit if it contained the reference entity under case-insensitive matching. Public
question answering used strict semantic scoring, which required the answer to satisfy
both factual and temporal conditions.
4.2 Overall Question Answering
With fact content and source order held constant, TEM-Status improved overall
key-entity hit rates by 17.14–55.95 percentage points over Unmarked across the four
backbones (Table 1).
Table 1Effect of historical labels on overall key-entity hit rate
Backbone Unmarked (%) TEM (%) Gain (pp)
Llama-3.1-8B 19.76 75.71 +55.95
Qwen3.5-4B 48.10 97.62 +49.52
Qwen3.5-9B 74.52 91.67 +17.14
gpt-oss-20b 55.24 96.67 +41.43
Each configuration contains 420 questions. Hit rates are percentages; gains are percentage points
(pp).
TEM-Status achieved overall key-entity hit rates of 75.71–97.62%, exceeding session-
aware RAG by 1.43–40.00 percentage points and full context by 2.14–30.24 points
(Table 2).
Table 2Overall key-entity hit rates across four backbones and six configurations (%)
Method Llama-3.1-8B Qwen3.5-4B Qwen3.5-9B gpt-oss-20b
Naive RAG 14.52 42.38 14.05 46.43
Session-aware RAG 56.67 57.62 79.76 95.24
Full context 49.76 95.48 87.38 66.43
Overwrite 43.33 29.76 41.43 34.76
Retain, unmarked 19.76 48.10 74.52 55.24
TEM-Status75.71 97.62 91.67 96.67
Each entry covers 420 questions. Bold indicates the highest value for a backbone.
4.3 Historical States after Repeated Updates
Overwrite recovered few historical residence values, with hit rates of 0–8.33%. Keeping
earlier versions without labels raised the rates to 41.67–81.67%. TEM-Status reached
96.67–100.00% after historical labels were added to the same evidence (Fig. 3a).
7

TEM-Status exceeded session-aware RAG by 10.00–88.33 percentage points and
full context by 3.33–40.00 points (Table 3).
Llama
8BQwen
4BQwen
9Bgpt-oss
20B0255075100Key-entity hit rate (%)
0.08.3
0.0 0.041.765.081.7
76.798.3 100.0 100.096.7a  Historical state retrieval
Overwrite Unmarked TEM-Status0 10 20 30 40 50 60
Gain over unmarked memory (pp)Llama-3.1-8B
Qwen3.5-4B
Qwen3.5-9B
gpt-oss-20b+55.95
+49.52
+17.14
+41.43b  Overall effect of state labels
Fig. 3Effects of version retention and historical labels. a, Historical-residence hit rates (60 questions).
b, Overall gains over Unmarked (420 questions). pp denotes percentage points.
Table 3Historical-residence retrieval after repeated updates (key-entity hit rate, %)
Method Llama-3.1-8B Qwen3.5-4B Qwen3.5-9B gpt-oss-20b
Naive RAG 0.00 0.00 0.00 5.00
Session-aware RAG 10.00 88.33 20.00 86.67
Full context 58.33 96.67 73.33 91.67
Overwrite 0.00 8.33 0.00 0.00
Retain, unmarked 41.67 65.00 81.67 76.67
TEM-Status98.33 100.00 100.00 96.67
Each entry covers 60 questions. Bold indicates the highest value for a backbone.
4.4 Ablation of State Labels
Adding historical labels gave 235 more hits with Llama-3.1-8B, 208 with Qwen3.5-4B,
72 with Qwen3.5-9B and 174 with gpt-oss-20b over 420 questions (Table 4). These
counts measure the net change in hits with fact content and source order held fixed.
Table 4Overall hit counts with and without historical labels
Backbone Unmarked hits TEM hits Net difference
Llama-3.1-8B 83 318 +235
Qwen3.5-4B 202 410 +208
Qwen3.5-9B 313 385 +72
gpt-oss-20b 232 406 +174
Each configuration contains 420 questions. Net gains are differences in total hit counts.
8

The selected examples illustrate errors in identifying the time or relation requested
by a question (Fig. 4). Under the control configurations, the same backbone sometimes
used an old value as current, confused versions around an update or returned an
unrelated relation. TEM recovered the requested value in each of these examples.
a  Current-state retrieval
Llama-3.1-8B
Q: Where does Gage live now?
Hangzhou
S1Wuhan
S4Beijing
S7
A · Naive RAG
"Wuhan." ×
A · TEM-Status
"Beijing."  ✓b  Historical-state assignment
Qwen3.5-4B
Q: Where did Nathan live before moving to
Beijing?
Shenzhen
S1Suzhou
S5Beijing
S8
A · Retain, unmarked
"Beijing" ×
A · TEM-Status
"Nathan lived in Suzhou before moving to
Beijing."  ✓
c  Historical-state selection
Qwen3.5-9B
Q: Where did Grace live before moving to
Hangzhou?
Tianjin
S1Chengdu
S4Hangzhou
S8
A · Session-aware RAG
"[...] Answer: Tianjin" ×
A · TEM-Status
"Grace lived in Chengdu before moving to
Hangzhou."  ✓d  Relation-specific recall
gpt-oss-20b
Q: What did Alice do before becoming a teacher?
doctor
S2teacher
S7
A · Session-aware RAG
"She switched to drinking water." ×
A · TEM-Status
"She was a doctor."  ✓
✓ correct answer   × incorrect answer   S: source session   [...] omitted text
Fig. 4Selected successful corrections across backbones. Arrows show source-session order; bold states
give the reference values. Check marks and crosses indicate correct and incorrect answers. The control
configuration is identified in each panel. Panel c marks omitted text with [. . . ].
Historical labels increased hit rates by 18.33–56.67 percentage points on the 60
residence questions (Table 5). The remaining 360 questions combine current-state
and other historical queries. Gains of 16.94–55.83 points on this group show that the
improvement extended beyond the historical-residence subset.
4.5 Answer-Evidence Length
TEM used 94.05–117.55 tokens of answer evidence on average, 48.75–56.73% less
than session-aware RAG (Table 6). These lengths exclude memory construction and
embedding computation.
TEM evidence ranged from 84 to 125 tokens across all queries (Fig. 5). Even the
shortest inputs from session-aware RAG and full context were longer, at 202 and 165
tokens.
9

Table 5Key-entity hit rates and label gains by query group
Backbone Other queries
Unmarked / TEMGain (pp) Pre-move residence
Unmarked / TEMGain (pp)
Llama-3.1-8B 16.11 / 71.94 +55.83 41.67 / 98.33 +56.67
Qwen3.5-4B 45.28 / 97.22 +51.94 65.00 / 100.00 +35.00
Qwen3.5-9B 73.33 / 90.28 +16.94 81.67 / 100.00 +18.33
gpt-oss-20b 51.67 / 96.67 +45.00 76.67 / 96.67 +20.00
Hit rates are percentages and gains are percentage points. Counts for other queries are obtained
by subtracting the historical-residence subset from the overall counts.
TEM-Status
Full context
Session-aware RAGa  Llama-3.1-8B b  Qwen3.5-4B
100 150 200 250TEM-Status
Full context
Session-aware RAGc  Qwen3.5-9B
100 150 200 250d  gpt-oss-20b
Evidence tokens per query
Fig. 5Answer-evidence length distributions, with 420 queries per configuration. Boxes show the
interquartile range, centre lines the median, and whiskers the minimum and maximum.
Table 6Mean evidence length during answering
Backbone Session-aware RAG Full context TEM-Status Saving (%)
Llama-3.1-8B 214.72 180.13 95.93 55.32
Qwen3.5-4B 229.36 199.52 99.25 56.73
Qwen3.5-9B 229.36 199.52 117.55 48.75
gpt-oss-20b 206.55 171.95 94.05 54.47
Evidence length is measured in tokens per question. Savings are relative to session-aware RAG.
4.6 Public Multi-Session Question Answering
We evaluated public question answering on 10 LoCoMo conversations containing 1,540
questions: 282 multi-hop, 321 temporal, 96 open-domain and 841 single-hop. Llama-
3.1-8B and Qwen3.5-9B each ran four configurations, producing 12,320 answers in
total. TEM-Hybrid-State, TEM-Hybrid-Unmarked and session-aware RAG shared a
1,024-token evidence limit. The two hybrid configurations used identical candidates
10

and ordering. A long-context configuration provided a further reference. All memories
were built from the source conversations only.
For each backbone, we kept the answer prompt, zero temperature, seed 42 and
1,024-token answer limit fixed across configurations. Qwen thinking was disabled.
Inputs to the long-context reference were truncated at the 32,768-token model window;
this affected 1,151 questions per backbone. We used gpt-oss-20b to score factual and
temporal correctness against the reference answers and supporting turns.
Ambiguity in the reference information left 216 answers unresolved. We retained
the full question count as the denominator and assigned these cases zero or one to
obtain score bounds. Eleven truncated or empty answers counted as incorrect.
Under the same retrieval budget, Llama scored 27.53–29.22% with TEM and
17.92–19.61% with session-aware RAG (Table 7). Qwen scored 35.84–37.53% and 23.70–
25.26%, respectively. Even subtracting the RAG upper bound from the TEM lower
bound left gaps of 7.92 and 10.58 percentage points.
Table 7Strict semantic scores on LoCoMo for two backbones (%)
Backbone Memory Overall Temporal Multi-hop Single-hop Open-
domain
Llama-3.1-
8BTEM-Hybrid-State 27.53–29.22 14.02–17.76 17.73–18.09 37.57–38.88 13.54–15.62
Llama-3.1-
8BTEM-Hybrid-
Unmarked27.40–28.96 14.33–17.45 14.89–15.25 38.17–39.48 13.54–15.62
Llama-3.1-
8BSession-aware RAG 17.92–19.61 8.72–11.84 12.06–12.77 23.90–25.33 13.54–15.62
Llama-3.1-
8BLong-context ref. 37.66–39.55 14.64–17.13 24.11–24.82 53.51–55.65 15.62–16.67
Qwen3.5-9B TEM-Hybrid-State 35.84–37.53 38.63–42.06 22.70–23.05 41.38–42.81 16.67–18.75
Qwen3.5-9B TEM-Hybrid-
Unmarked35.97–37.66 38.32–41.43 25.18–25.89 40.78–42.21 17.71–19.79
Qwen3.5-9B Session-aware RAG 23.70–25.26 23.99–27.10 19.86–20.57 25.92–27.23 14.58–15.62
Qwen3.5-9B Long-context ref. 53.90–56.17 48.91–52.96 28.72–30.14 66.94–68.85 30.21–32.29
Each configuration contains 1,540 questions. Ranges are bounds obtained by assigning unresolved
scores zero or one, not confidence intervals. The first three configurations have a 1,024-token
evidence limit; the long-context reference has a 32,768-token window.
Historical labels made little difference to overall LoCoMo scores. Mean TEM
evidence lengths were 1,014.6 tokens for Llama and 1,017.5 for Qwen, close to session-
aware RAG. The long-context reference scored higher with much larger inputs: 29,671.8
and 30,412.9 tokens, respectively. These public questions require event details and
multi-hop associations as well as state information. The similar scores of the marked
and unmarked variants suggest that hybrid evidence organisation accounts for most of
the observed gain over RAG.
4.7 Results Reported for Existing Memory Systems
Published memory-system evaluations provide further context for these results [ 10–13].
Their generators and scoring metrics are recorded alongside the reported scores in
Table 8. TEM was evaluated with open-weight generators and strict semantic scoring.
11

Table 8Published results and evaluation settings of representative memory methods
Method Generator Benchmark Metric Overall Temporal Single-hop
A-MEM [10] GPT-4o-mini LoCoMo F1 — 45.85 44.65
MemoryOS [12] GPT-4o-mini LoCoMo F1 — 20.02 35.27
Mem0 [11] GPT-4o-mini LoCoMo J 66.88±0.15 55.51±0.34 67.13±0.65
Mem0-graph
[11]GPT-4o-mini LoCoMo J 68.44±0.17 58.13±0.44 65.71±0.45
Zep [13] GPT-4o-mini LongMemEval-s GPT-4o
judge63.80 54.10 —
Zep [13] GPT-4o LongMemEval-s GPT-4o
judge71.20 62.40 —
TEM-Hybrid Llama-3.1-8B LoCoMo Strict
judge27.53–29.22 14.02–17.76 37.57–38.88
TEM-Hybrid Qwen3.5-9B LoCoMo Strict
judge35.84–37.53 38.63–42.06 41.38–42.81
Scores are percentages. Generators, scoring rules and input budgets differ, so values are not a
common ranking. A-MEM values are from v11, Table 1; MemoryOS values are from Table 2 of
the published version. Mem0 reports standard deviations over 10 evaluations. TEM ranges bound
unresolved scores. A dash indicates an unreported or inapplicable value.
5 Discussion
The state-update results separate two requirements of historical question answering. A
memory must first keep earlier values accessible. Once those values are available, the
generator still needs to identify which one applies. In the ablation, historical labels
improved hit rates without changing the facts or their source order. Retrieval therefore
did not account for the difference between the two configurations. The result supports
treating version status as part of the evidence presented to the generator, even when it
can in principle be inferred from the ordered records.
On LoCoMo, marked and unmarked hybrid memory obtained similar scores, and
both exceeded session-aware RAG under the same evidence budget. The contrast with
the state-update task helps define the role of the labels. They directly indicate whether
a retained value has been superseded, but do not supply missing event details or
cross-session associations. These are also needed for public dialogue questions. The
hybrid-memory gain on LoCoMo is therefore better understood at the level of evidence
organisation, without attributing it to historical labels alone.
TEM uses source-session order to assign state, assuming explicit single-valued
updates. This is sufficient for the controlled update setting considered here. A delayed
report may place an earlier event in a later session, and contradictory values can occur
within one session. Preferences may also remain valid simultaneously. Handling these
cases requires event-validity times or explicit treatment of conflicting relations beyond
the present ordering rule.
6 Conclusion
TEM couples history-preserving fact storage with evidence that makes version status
explicit. The update rule operates on complete version groups, and retrieval passes
source order and historical labels to the generator. Across four backbones, retaining
12

and labelling versions improved historical information use in state-update tasks with
compact fact evidence. Hybrid memory also exceeded session-aware RAG on public
multi-session questions at the same evidence budget. Together, the results support
managing fact versions throughout storage and answering when information changes
across conversations.
Statements and Declarations
Funding
This research was supported by Zhejiang Provincial Natural Science Foundation of
China under Grant No. LQN26F020016.
Competing Interests
The authors declare that they have no competing interests.
Data and Code Availability
The public LoCoMo dataset used in this study is available from its official source [ 24].
Illustrative code for the TEM mechanism and an overview of the main results are
available at https://github.com/sungkites/TEM.
References
[1]Zhang, Z., Dai, Q., Bo, X.,et al.: A Survey on the Memory Mechanism of Large
Language Model-based Agents. ACM Transactions on Information Systems43(6),
1–47 (2025) https://doi.org/10.1145/3748302
[2]Hu, Y., Liu, S., Yue, Y., et al.: Memory in the Age of AI Agents. arXiv preprint
arXiv:2512.13564 (2025)
[3]Wang, L., Ma, C., Feng, X.,et al.: A survey on large language model based
autonomous agents. Frontiers of Computer Science18(6), 186345 (2024) https:
//doi.org/10.1007/s11704-024-40231-1
[4]Park, J.S., O’Brien, J., Cai, C.J.,et al.: Generative Agents: Interactive Simulacra
of Human Behavior. In: Proceedings of the 36th Annual ACM Symposium on
User Interface Software and Technology, pp. 1–22 (2023). https://doi.org/10.1145/
3586183.3606763
[5]Yao, S., Zhao, J., Yu, D., et al.: ReAct: Synergizing Reasoning and Acting in
Language Models. arXiv preprint arXiv:2210.03629 (2022)
[6]Shinn, N., Cassano, F., Berman, E., et al.: Reflexion: Language Agents with Verbal
Reinforcement Learning. arXiv preprint arXiv:2303.11366 (2023)
13

[7]Packer, C., Wooders, S., Lin, K., et al.: MemGPT: Towards LLMs as Operating
Systems. arXiv preprint arXiv:2310.08560 (2023)
[8]Zhong, W., Guo, L., Gao, Q.,et al.: MemoryBank: Enhancing Large Language
Models with Long-Term Memory. In: Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 38, pp. 19724–19731 (2024). https://doi.org/10.1609/
aaai.v38i17.29946
[9]Wang, W., Dong, L., Cheng, H.,et al.: Augmenting Language Models with Long-
Term Memory. In: Advances in Neural Information Processing Systems, vol. 36,
pp. 74530–74543 (2023). https://doi.org/10.52202/075280-3259
[10]Xu, W., Liang, Z., Mei, K., et al.: A-MEM: Agentic Memory for LLM Agents.
arXiv preprint arXiv:2502.12110 (2025)
[11]Chhikara, P., Khant, D., Aryan, S., et al.: Mem0: Building Production-Ready AI
Agents with Scalable Long-Term Memory. arXiv preprint arXiv:2504.19413 (2025)
[12]Kang, J., Ji, M., Zhao, Z.,et al.: Memory OS of AI Agent. In: Proceedings of
the 2025 Conference on Empirical Methods in Natural Language Processing, pp.
25961–25970 (2025). https://doi.org/10.18653/v1/2025.emnlp-main.1318
[13]Rasmussen, P., Paliychuk, P., Beauvais, T., et al.: Zep: A Temporal Knowledge
Graph Architecture for Agent Memory. arXiv preprint arXiv:2501.13956 (2025)
[14]Fang, J., Deng, X., Xu, H., et al.: LightMem: Lightweight and Efficient Memory-
Augmented Generation. arXiv preprint arXiv:2510.18866 (2025)
[15]Guti´ errez, B., Shu, Y., Gu, Y.,et al.: HippoRAG: Neurobiologically Inspired Long-
Term Memory for Large Language Models. In: Advances in Neural Information
Processing Systems, vol. 37, pp. 59532–59569 (2024). https://doi.org/10.52202/
079017-1902
[16]Guti´ errez, B.J., Shu, Y., Qi, W., et al.: From RAG to Memory: Non-Parametric
Continual Learning for Large Language Models. arXiv preprint arXiv:2502.14802
(2025)
[17]Edge, D., Trinh, H., Cheng, N., et al.: From Local to Global: A Graph RAG
Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130
(2024)
[18]Qian, H., Liu, Z., Zhang, P., et al.: MemoRAG: Boosting Long Context Pro-
cessing with Global Memory-Enhanced Retrieval Augmentation. arXiv preprint
arXiv:2409.05591 (2024)
[19]Li, Z., Xi, C., Li, C., et al.: MemOS: A Memory OS for AI System. arXiv preprint
arXiv:2507.03724 (2025)
14

[20]Linders, J., Tomczak, J.M.: Knowledge graph-extended retrieval augmented
generation for question answering. Applied Intelligence55(17), 1102 (2025)
https://doi.org/10.1007/s10489-025-06885-5
[21]Jiao, S., Zhu, Z., Wu, W.,et al.: An improving reasoning network for complex
question answering over temporal knowledge graphs. Applied Intelligence53(7),
8195–8208 (2023) https://doi.org/10.1007/s10489-022-03913-6
[22]Fang, J., Lu, L., Liu, X.: MGIT: Multi-Granularity implicit temporal framework
for knowledge graph question answering. Applied Intelligence56(12), 374 (2026)
https://doi.org/10.1007/s10489-026-07404-w
[23]Wu, D., Wang, H., Yu, W., et al.: LongMemEval: Benchmarking Chat Assistants
on Long-Term Interactive Memory. arXiv preprint arXiv:2410.10813 (2024)
[24]Maharana, A., Lee, D.-H., Tulyakov, S.,et al.: Evaluating Very Long-Term
Conversational Memory of LLM Agents. In: Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 13851–13870 (2024). https://doi.org/10.18653/v1/2024.acl-long.747
[25]Bai, Y., Lv, X., Zhang, J.,et al.: LongBench: A Bilingual, Multitask Benchmark
for Long Context Understanding. In: Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
3119–3137 (2024). https://doi.org/10.18653/v1/2024.acl-long.172
[26]Liu, N.F., Lin, K., Hewitt, J.,et al.: Lost in the Middle: How Language Models
Use Long Contexts. Transactions of the Association for Computational Linguistics
12, 157–173 (2024) https://doi.org/10.1162/tacl a00638
[27]Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. arXiv preprint arXiv:2005.11401 (2020)
[28]Karpukhin, V., Oguz, B., Min, S.,et al.: Dense Passage Retrieval for Open-
Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP), pp. 6769–6781 (2020). https:
//doi.org/10.18653/v1/2020.emnlp-main.550
[29]Asai, A., Wu, Z., Wang, Y., et al.: Self-RAG: Learning to Retrieve, Generate, and
Critique through Self-Reflection. arXiv preprint arXiv:2310.11511 (2023)
[30]Sarthi, P., Abdullah, S., Tuli, A., et al.: RAPTOR: Recursive Abstractive
Processing for Tree-Organized Retrieval. arXiv preprint arXiv:2401.18059 (2024)
[31]Trivedi, H., Balasubramanian, N., Khot, T.,et al.: Interleaving Retrieval with
Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In:
Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 10014–10037 (2023). https://doi.org/10.
15

18653/v1/2023.acl-long.557
[32]Jiang, H., Wu, Q., Lin, C.-Y.,et al.: LLMLingua: Compressing Prompts for
Accelerated Inference of Large Language Models. In: Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pp. 13358–
13376 (2023). https://doi.org/10.18653/v1/2023.emnlp-main.825
[33]Jiang, H., Wu, Q., Luo, X.,et al.: LongLLMLingua: Accelerating and Enhancing
LLMs in Long Context Scenarios via Prompt Compression. In: Proceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 1658–1677 (2024). https://doi.org/10.18653/v1/2024.acl-long.
91
[34]Xu, F., Shi, W., Choi, E.: RECOMP: Improving Retrieval-Augmented LMs with
Compression and Selective Augmentation. arXiv preprint arXiv:2310.04408 (2023)
16