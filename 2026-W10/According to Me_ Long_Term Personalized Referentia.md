# According to Me: Long-Term Personalized Referential Memory QA

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Xinyu Hou, Margaret Li, Bill Byrne

**Published**: 2026-03-02 15:42:29

**PDF URL**: [https://arxiv.org/pdf/2603.01990v1](https://arxiv.org/pdf/2603.01990v1)

## Abstract
Personalized AI assistants must recall and reason over long-term user memory, which naturally spans multiple modalities and sources such as images, videos, and emails. However, existing Long-term Memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. We introduce ATM-Bench, the first benchmark for multimodal, multi-source personalized referential Memory QA. ATM-Bench contains approximately four years of privacy-preserving personal memory data and human-annotated question-answer pairs with ground-truth memory evidence, including queries that require resolving personal references, multi-evidence reasoning from multi-source and handling conflicting evidence. We propose Schema-Guided Memory (SGM) to structurally represent memory items originated from different sources. In experiments, we implement 5 state-of-the-art memory systems along with a standard RAG baseline and evaluate variants with different memory ingestion, retrieval, and answer generation techniques. We find poor performance (under 20\% accuracy) on the ATM-Bench-Hard set, and that SGM improves performance over Descriptive Memory commonly adopted in prior works. Code available at: https://github.com/JingbiaoMei/ATM-Bench

## Full Text


<!-- PDF content starts -->

According to Me: Long-Term Personalized Referential Memory QA
Jingbiao Mei1Jinghong Chen1Guangyu Yang1Xinyu Hou2Margaret Li3Bill Byrne1
Abstract
Personalized AI assistants must recall and rea-
son over long-term user memory, which naturally
spans multiple modalities and sources such as
images, videos, and emails. However, existing
Long-term Memory benchmarks focus primarily
on dialogue history, failing to capture realistic
personalized references grounded in lived experi-
ence. We introduce ATM-Bench, the first bench-
mark for multimodal, multi-source personalized
referential Memory QA. ATM-Bench contains
approximately four years of privacy-preserving
personal memory data and human-annotated ques-
tion–answer pairs with ground-truth memory evi-
dence, including queries that require resolving
personal references, multi-evidence reasoning
from multi-source and handling conflicting ev-
idence. We propose Schema-Guided Memory
(SGM) to structurally represent memory items
originated from different sources. In experiments,
we implement 5 state-of-the-art memory systems
along with a standard RAG baseline and evalu-
ate variants with different memory ingestion, re-
trieval, and answer generation techniques. We
find poor performance (under 20% accuracy) on
the ATM-Bench-Hard set, and that SGM im-
proves performance over Descriptive Memory
commonly adopted in prior works. Code available
at: https://github.com/JingbiaoMei/ATM-Bench.
1. Introduction
Human Memory is distinctively diverse (Harvey et al., 2016).
Real memory derives from data of multiple modalities (text,
vision) and many sources (photos, e-mail, etc.). Yet cur-
rent benchmarks onMemory-based Question-Answering
(Memory QA)typically focus on conversational history
alone (Zhong et al., 2024; Maharana et al., 2024; Du et al.,
1Department of Engineering, University of Cambridge,
United Kingdom2Department of Physics, University of Cam-
bridge, United Kingdom3Independent Researcher. Correspon-
dence to: Jingbiao Mei <jm2245@cam.ac.uk >, Bill Byrne
<wjb31@cam.ac.uk>.
Preprint. March 3, 2026.2024; Wu et al., 2025). This overlooks informative static
sources to capture the user’s lived experience. For example,
a photo of a concert ticket (with explicit user permission)
could provide information about the location, time, and
content of the event without user engagement. To develop
memory agents that can garner the full range of memory
data, a memory benchmark that contains multimodal, multi-
sourced data is required.
Human Memory is also distinctively personalized referen-
tial (Du et al., 2024; Maharana et al., 2024; Tan et al., 2025).
We refer to things in the context of our own experience. For
example, resolving “the gift I bought for mum during our
trip to Japan” can be very useful for real-world memory
agent application, but would require a holistic understand-
ing of one’s personal experience. To our knowledge, current
Memory QA benchmarks fall short of the real-world com-
plexity in terms of personalized references. It is therefore
unclear how well state-of-the-art systems perform on these
realistic and challenging queries.
In this work, we propose ATM-Bench, the first benchmark
for multimodal, multi-source personalized referential Mem-
ory QA. ATM-Bench comprises approximately four years
of privacy-preserving personal memory data spanning mul-
tiple countries and continents, covering diverse everyday do-
mains (e.g., travel, social events, work, and arts), including
emails, images, and videos, with explicit temporal and spa-
tial grounding. We collect more than 1,000 question–answer
pairs and provide human-annotated ground-truth memory
items. The set contains challenging queries that require re-
solving personal references, handling conflicting evidence,
and locating memory items based on temporal and visual
cues (Figure 1). We evaluate a range of state-of-the-art long-
term memory systems and find unsatisfactory performance.
Notably, on the curated hard set (ATM-Bench-Hard), the
current best system falls short of 20% accuracy. We believe
this opens up venues for future Memory QA research.
We also develop formalism and methodology to handle
multi-sourced memory. Specifically, we formulate the Mem-
ory QA task into three steps: (1)Memory Ingestionwhich
processes raw data and constructs the memory base for
retrieval. Designing Memory Ingestion pipeline typically
involves choosing Memory Representation (the format of
a memory item) and Memory Organization (how to repre-
1arXiv:2603.01990v1  [cs.AI]  2 Mar 2026

Long-Term Personalized Referential Memory QA
Figure 1.Core challenges in Long-term Multi-source Multimodal Personalized Referential Memory QA.(A) Resolving Personal
References.Given a multimodal personal database, the agent needs to resolve the user’s reference (“Grace”) by linking to the user’s
past memory, and subsequently retrieve moments matching the query’s condition (“being sneaky”).(B) Conflict-aware Multi-source
Aggregation with Memory Updates Over Time.To compute total hotel expenses, the model aggregates heterogeneous evidence
(emails and receipts), resolves conflicts between booking records and finalized invoices, and prioritizes later-updated memory entries
that reflect the most recent and finalized information. This evaluates the agent’s ability to handle memory updates over time.(C)
Temporal–Location–Visual Grounding.When images lack precise location cues, the model infers an event time window from an email
memory item, retrieves images within that interval, and grounds them to the queried location to recover the ordered set of memory items.
sent relations between memory items); (2)Retrievalwhich
retrieves the relevant memory items given a query; and (3)
Question Answeringwhich answers the query given the re-
trieved memory items. We find that this formalism well
describes current Memory QA approaches and provides a
framework for comparing system designs. To handle data
from heterogeneous sources and to include useful meta-data
such as timestamps and GPS location, we propose Schema-
Guided Memory (SGM), a structured Memory Represen-
tation method. Our experiments show SGM substantially
improves both QA and retrieval performance compared to
Descriptive Memory commonly adopted by prior works.
Our contributions are summarized as follows:
1.Data.We introduce ATM-Bench, the first Memory
QA benchmark designed to evaluate personalized AI
agents on long-term, multimodal, and multi-sourced
memory that contains realistically complex personal-
ized references. ATM-Bench comprises 1038 ques-
tion–answer pairs with cross-validated ground-truth
memory evidence to enable precise evaluation of mem-
ory grounding1.
1The benchmark corpus and evaluation suite will be released2.Problem Formulation.We formulate the problem of
Memory QA into Memory Ingestion, Retrieval, and
Answer Generation (Sec. 4). This formalism allows us
to compare variants of current memory systems and
study the effectiveness of different designs.
3.Schema-Guided Memory (SGM).We propose a struc-
tured format to represent memory items that could orig-
inate from different sources and encompass different
modalities (e.g., gallery and email). We find that SGM
improves performance compared to the Descriptive
Memory (DM) paradigm adopted in most prior works.
4.Analysis of current memory systemsWe imple-
ment and evaluate 5 state-of-the-art memory systems
(Sec. 5). Our analysis reveals a critical performance
gap: current methods attain under 20% accuracy on
the challenging ATM-Bench-Hard set. We also reports
the effect of different Memory Representation (SGM
vs. DM) and Memory Organization methods (Piled vs.
Linked, Sec. 4.2).
upon publication.
2

Long-Term Personalized Referential Memory QA
2. Related Work
2.1. Long-term Memory Benchmark
To support personalized assistants, models must recall user-
specific long-term context such as preferences and inter-
action history (Li et al., 2024). While recent long-context
LLMs significantly extend accessible sequence lengths (Liu
et al., 2025), scaling context alone does not solve long-term
recall (Hosseini et al., 2025). In practice, long-context mod-
els exhibit context forgetting (Liu et al., 2024), which moti-
vates benchmarks that explicitly evaluate episodic memory
retrieval and evidence-grounded reasoning beyond long-
context prompting (Maharana et al., 2024; Wu et al., 2025).
Most existing long-term memory benchmarks are text-
centric and derived from human–assistant dialogues, evalu-
ating memory via QA or related conversational tasks (Tan
et al., 2025). LongMemEval (Wu et al., 2025) assesses mul-
tiple long-term memory abilities over attribute-controlled
interaction histories, while LoCoMo (Maharana et al., 2024)
evaluates very long-term conversational memory across
multi-session dialogues. MemoryBank (Zhong et al., 2024)
studies personalized dialogue with external memory mod-
ules, and PerLTQA (Du et al., 2024) formulates personal
long-term memory as QA over mixed semantic and episodic
text memories. More recent work explores multimodal mem-
ory. OmniQuery (Li et al., 2025) studies free-form QA
over user-captured media through human evaluation, while
Memory-QA (Jiang et al., 2025) targets recall-oriented QA
grounded in stored visual memories. In contrast, ATM-
BENCHfocuses onmulti-source, multimodalpersonal mem-
ory with explicit temporal and spatial grounding, enabling
evaluation of evidence aggregation and referential reasoning
across heterogeneous personal artifacts (e.g., jointly resolv-
ing queries over images, emails, and metadata). A more
detailed comparison with prior benchmark is provided in
Table 1 and Sec. 3.2.
2.2. Memory System
To address long-term memory benchmarks, recent work
augments LLM agents with explicit memory management,
typically treating the dialogue history as the primary and of-
ten single-source memory trace. Frameworks like MemGPT
(Packer et al., 2024) and MemoryOS (Kang et al., 2025)
abstract this as an operating system challenge, managing
virtual context paging. Mem0 (Chhikara et al., 2025) and
Memobase (MemoDB, 2024) decouple storage from infer-
ence to improve scalability. A-Mem (A-MEM) (Xu et al.,
2025) constructs structured notes and dynamically links
them into an interconnected graph. Beyond external storage,
architectural extensions like M+ (Wang et al., 2025) attempt
to integrate long-term memory directly into model weights.2.3. Retrieval-Augmented QA
Retrieval-Augmented Generation (RAG) (Lewis et al., 2020)
is a standard approach for knowledge-intensive generation
and open-domain QA, using a passage retriever (Karpukhin
et al., 2020) to support multi-hop evidence needs in datasets
such as HotpotQA (Yang et al., 2018).
RAG has evolved from a single pass retrieve-then-answer
(Izacard et al., 2022) into iterative, agentic workflows. En-
hancements include optimized query decomposition (Press
et al., 2023; Jain et al., 2025) and mechanisms for self-
reflection (Asai et al., 2024; Shao et al., 2024).
Despite these advances, personalized referential reasoning
differs from standard QA as queries are often underspeci-
fied and rely on implicit cues; separating what to retrieve
from how to reason remains intrinsically hard (Jin et al.,
2025). Existing evaluations largely stay within a single
source dialogue histories (Maharana et al., 2024; Wu et al.,
2025) or media-centric recall stores in Memory-QA (Jiang
et al., 2025). Our work addresses this gap by evaluating
multi-source retrieval and reasoning that requires aggregat-
ing evidence across heterogeneous personal memory data
with explicit temporal and location grounding.
3. ATM-Bench
3.1. Problem Formulation
ATM-Bench formalizesLong-term Personalized Referen-
tial Memory QAas the task of retrieving and reasoning
over a user’s long-term, multimodal, and multi-source per-
sonal memories to answer context-dependent queries. ATM-
Bench consists of a multimodal personal memory corpus
D, together with a set of question qi, ground truth answer
ai, and evidence set Eifor each instance i. The raw data of
ATM-BenchDare given by:
D={I i}NI
i=1∪ {V i}NV
i=1∪ {E i}NE
i=1.(1)
where{Ii},{Vi}, and{Ei}denote image, video, and email
raw memory data, respectively.
3.2. Characteristics of ATM-Bench
The personal memory data used in ATM-Bench were col-
lected over a period of approximately four years by human.
The questions, answers and evidence in ATM-Bench are
fully annotated by humans, details including why agentic
auto-annotation failed are provided in Appendix B.2. Eth-
ical considerations and risk mitigation are the priority in
designing ATM-Bench; a comprehensive ethical assessment
is provided in Appendix A.
ATM-Bench is designed to capture challenges intrinsic to
Long-term Personalized Referential Memory QA that are
not adequately addressed by existing benchmarks. In par-
3

Long-Term Personalized Referential Memory QA
Table 1.Comparison of ATM-Bench with other memory and long-context QA benchmarks.PR: Personalized Referential Reasoning;LA:
Location Awareness;MUT: Memory Update Over Time;ME: Multi-Evidence Composition (multi-session reasoning for dialogue-based
benchmarks);ABS: Abstention. For Memory-QA, we report the largertest-lsplit.#Memdenotes the number of memory items, interpreted
as the average number of sessions for dialogue datasets.Contextdenotes the number of tokens accessible to the agent. Media data are
counted as 300 tokens each.
Benchmark Domain Mem Origin Data Source #Mem #Q Time Span Context Required Memory Abilities
PR LA MUT ME ABS
MemoryBank Personal Dialogue Synthetic 300 194 Days 5k × × × × ×
PerLTQA Personal Dialogue Synthetic∼4.5k∼8.6k Years 1M✓× × ×✓
LoCoMo Personal Dialogue Hybrid∼1k 7.5k Months 10k✓× ×✓ ✓
DialSim TV Shows Dialogue Hybrid∼1.3k 1M Years 350k × × ×✓ ✓
LongMemEval Personal Dialogue Hybrid 500 500 Months 1.5M × ×✓ ✓ ✓
Memory-QA Personal Img Hybrid 5.8k∼9.3k Months 840k ×✓×✓×
ATM-Bench Personal Img, Vid, Emails Human 12k 1.1k Years 2.25M✓ ✓ ✓ ✓ ✓
ticular, it emphasizes the following characteristics, aligned
with the core memory abilities summarized in Table 1.
Personalized Referential Reasoning (PR).Questions in
ATM-Bench frequently rely on implicit, user-specific ref-
erences rather than direct entity mentions. Events, people,
or objects are often referred to through personal context
(e.g., “the trip celebrating my birthday”), requiring systems
to ground queries in personal memory histories.
Location Awareness (LA).ATM-Bench features many
queries that requires identifying where an event occurred,
leveraging location metadata such as GPS coordinates, place
names, or locations inferred from visual cues in photos and
textual information in emails. This location awareness is
essential for distinguishing similar events across different
locations and for resolving place-based references over long
time spans.
Memory Update Over Time (MUT).ATM-Bench contains
examples where explicit memory updates over time are
necessary as newly acquired memories may revise, override,
or refine earlier beliefs (e.g., updated travel plans, corrected
information, or evolving user preferences). This requires
more than append-only memory accumulation and requires
a memory agent to reason over memory update.
Multi-Evidence Composition (ME).Answering a question
often requires aggregating evidence from multiple memory
items, spanning different modalities and sources such as
images, videos, and emails. Individual memory items may
be weakly informative in isolation but jointly sufficient when
composed for answering the question, demanding robust
multi-hop and cross-modal retrieval and reasoning.
Abstention (ABS).ATM-Bench includes queries that are
unanswerable given all the available memory evidence.
Models are expected to recognize insufficient or missing
information and abstain from answering, rather than halluci-
nating unsupported responses.
Comparison with Prior Work.EstablishingPRin ATM-Bench is much more challenging than in dialogue-based
memory datasets. In dialogue settings, PR is usually es-
tablished by the user explicitly informing the agent in a
conversation. In contrast, in ATM-Bench, PR cues are im-
plicitly distributed across heterogeneous modalities. For
example, the name of the pet may be discovered from email
(See Figure 1(A)).
While Memory-QA is one of the few prior benchmarks
that combinesLA, ATM-Bench integrates location sig-
nals from both emails and visually recognized content
(see Figure 1(C)), resulting in substantially broader and
more diverse multi-continental geographic coverage (Ap-
pendix C.2). In addition, approximately 30% of ATM-
Bench questions requireME, a substantially higher propor-
tion than in prior benchmarks such as Memory-QA (Jiang
et al., 2025), where only 6% of questions involve multiple
evidence items.
ATM-Bench also models memory evolution over time in a
more realistic manner, introducing a particularly challeng-
ingMUTsetting (see Figure 1(B)). Our analysis shows
that even state-of-the-art models operating under gold-
evidence conditions consistently fail to answer correctly
(Appendix F). Furthermore, the time span between the ear-
liest and latest evidence items required to answer a single
question,averages 226 dayson the hard split, with a maxi-
mum span of 933 days.
3.3. ATM-Bench Statistics
Table 2 shows the distribution of the memory corpus across
different modalities. Token counts are computed over the
full textual metadata associated with each memory data,
including timestamps. For images and videos, we report the
resolution and format.
The QA dataset is divided into two subsets: ATM-Bench,
with an average of 1.6 evidence items per query, and ATM-
Bench-Hard, which requires an average of6.3 evidence
4

Long-Term Personalized Referential Memory QA
Table 2.Corpus statistics. Tokens are reported as mean±std.
Modality Count Token/Resolution
Email 6,741 101±22
Image 3,759 2MegaPixels
Video 533 480p
itemsdrawn from multiple sources.
More detailed statistics and analyses of the memory corpus
and benchmark questions, including memory semantic clas-
sification, question-type classification, evidence counts, and
temporal spans, are provided in Appendix C.
3.4. Evaluation Metrics
We evaluate personalized Memory QA systems on question
answering and memory retrieval.
Answer EvaluationTo account for the diverse answer
structures in personal memory questions, we introduce the
Question Type Score(QS), an answer-type–aware evalua-
tion framework. In addition to the reference answer a, we
annotate one of three answer types tfor each question: (1)
Number, which includes scalar answers like dates, times,
durations, or currency values; (2) List Recall, which re-
quires enumerating multiple items such as entities, or mem-
ory items; and (3) Open-ended, which involves free-form
natural-language responses requiring descriptive or explana-
tory content.
QS(q, a,ˆa) =

EM(a,ˆa),ift=number,
J(a,ˆa),ift=list recall,
LLM-Judge(q, a,ˆa),ift=open-ended,
(2)
J(·,·)is the Jaccard similarity:
J(a,ˆa) =|toList(a)∩toList(ˆa)|
|toList(a)∪toList(ˆa)|.(3)
Here, EM (Exact Match) and LLM-Judge produce binary
scores in {0,1} , while J(a,ˆa) yields a real-valued score in
[0,1] . For EM, answers are post-processed similar to that in
VQA (Mensink et al., 2023; Chen et al., 2023).
Memory Retrieval EvaluationWe reportRecall@k, de-
fined as the fraction of ground-truth evidence retrieved
among the top- kresults, and Recall@k GT, with kset to
the number of ground-truth evidence items.
To assess whether systems answer questions using the cor-
rect memory evidence, we further report a joint metric that
combines answer correctness with retrieval quality (Yang
et al., 2018), for each item:
Joint@k=QS(q, a,ˆa)×Recall@k.(4)4. Memory Assistant Methodology
4.1. Personal Memory Assistant Formulation
In this section, we formulate a multi-source, multimodal per-
sonal memory assistant for question answering. As shown
in Figure 2 the system has three steps: (i)Memory Ingestion
that converts heterogeneous personal artifacts into a nor-
malized memory store via preprocessing and organization,
(ii)retrievalthat selects a small evidence set given a user
query, and (iii)answer generationthat generates a grounded
response from retrieved evidence.
Figure 2.Personal Memory Assistant Overview.
To obtain a unified representation from the raw personal
memory data D(Eq. 1), we apply a preprocessing function
that transforms the heterogeneous corpus into a memory
representation:
M={m i}N
i=1=P(D).(5)
5

Long-Term Personalized Referential Memory QA
Optionally, additional organization operators may be ap-
plied to M, such as constructing graph-based linkages and
updating memory items to across sources.
Given a user query q, a retriever Rretrieve the top- kmost
relevant memory items from the memory store:
E=R(q,M),(6)
where E ⊆ M denotes the retrieved evidence set. An
answerer Athen generates an answer conditioned on the
query and retrieved evidence:
ˆa=A(q,E).(7)
4.2. Memory Ingestion
We split the Memory Ingestion into preprocessing and op-
tional organization.
Memory Preprocessing.We consider two types of prepro-
cessed memory representation: Descriptive Memory (DM)
and Schema-Guided Memory (SGM).
DMrepresents each memory item as a natural-language
description,
MD=PD(D), mD
i∈Text,(8)
where each mD
iis a natural-language text sequence. For
example “ image2020010115000, On Jan 1st,
2020 afternoon, I was at Scotiabank
Arena, watching ice hockey match...”.
SGMrepresents each memory item as a set of text-based
key–value fields defined under a fixed schemaS.
MS=PS(D), m i={(k, v i,k)|k∈ S},(9)
where Sincludes fields such as time ,location ,
entities ,OCR, and tags . For email memories, Sin-
cludes fields such as time ,summary , andbody . A SGM
may take the form:
{id : "image2020010115000" ,time :
"2020-01-01 15:00" , location :
"Scotiabank Arena" ,source :"image" ,
entities : [ "Scotiabank Arena" ,"ice
hockey" ],tags : ["sports" ,"event" ],OCR:
"Scotiabank"}
Notably, SGM and DM encompass the same information
and differ only in how this information is formatted.
Memory OrganizationBeyond per-item representation,
an agent could organize memory items into structures, such
as graphs that link related items. We refer to memory stored
without explicit structure as Piled Memory. In contrast,Linked Memory introduces explicit relations among mem-
ory items.
Formally, Linked Memory is constructed via a linking func-
tion that computes the adjacency matrix between memory
items.
LΘ:M →RN×N(10)
The resulting structure is a graph of memory items, in which
the links are inferred by LLM Θand can be exploited during
retrieval.
Furthermore, agentic memory systems like A-Mem (Xu
et al., 2025), allow memory items to be updated during the
organization process. We define memory update function
FΘ:M → M,(11)
which modifies existing memory items based on newly in-
ferred relations by LLMΘ.
4.3. Retrieval
Retriever R(Eq. 6) uses embedding-based similarity search.
Specifically, both the user query qand memory items
m∈ M are encoded into a shared vector space using a
pretrained text or multimodal embedding model. Retrieval
is then performed by selecting the top- kmemory items with
the highest similarity scores to the query, measured via
maximum inner product search (MIPS).
Formally, let ϕq(q)andϕm(m) denote the embedding func-
tions for the query and memory items, respectively. The
retrieverRreturns
E= TopKm∈M⟨ϕq(q), ϕ m(m)⟩(12)
where⟨·,·⟩denotes inner product.
4.4. Answer Generation
We compare two answer generation paradigms. A single-
pass answerer directly generates the final answer ˆafrom the
query–evidence pair (q,E) in a single forward pass. In con-
trast, anagentic answererperforms iterative reasoning: it
plans and decomposes the query, rewriting query to perform
additional retrieval when the available evidence is insuffi-
cient before producing the final answer.
5. Experiments
We evaluate baselines on ATM-Bench and the more chal-
lenging ATM-Bench-Hard, and analyze the impact of design
choices across Memory Ingestion, retrieval, and answering.
5.1. Experimental Setup
Retrieval.we use all-MiniLM-L6-v2 as the default re-
triever embedding model (Reimers & Gurevych, 2019;
6

Long-Term Personalized Referential Memory QA
Table 3.Comparison of memory systems.Rep.denotes memory representation andOrg.denotes memory organization. Tencdenotes
memory encoding time (in hours). QS is the overall metric.N/R/Odenote accuracy for Number, Recall-list, and Open-ended questions.
The 0.2% of QS score of No-Evidence corresponds to the 0.3% of the Abstention questions in the dataset.
# System Memory ATM-Bench ATM-Bench-Hard
Rep. Org.T enc QS R@10 Joint@10 N R O QS R@10 Joint@10
Upper / Lower Bounds
1 No-Evidence – – – 0.2 – – 0.0 0.0 0.6 0.0 – –
2 Oracle DM – – 70.0 – – 81.8 69.3 61.9 25.6 – –
3 Oracle SGM – – 77.8 – – 85.0 90.4 69.5 47.3 – –
Memory Agents
4 A-Mem DM Piled 1.2 46.1 66.6 44.0 55.0 27.0 44.9 15.038.412.8
5 A-Mem DM Linked 12.6 44.8 66.4 42.8 57.2 16.9 43.6 10.0 37.8 9.5
6 Mem0 Agentic DM Linked 16.7 43.5 61.9 41.8 57.2 25.9 48.4 16.5 37.7 16.0
7 Mem0 Plain DM Linked 16.7 41.5 61.9 38.4 58.6 25.9 33.7 16.6 37.7 15.7
RAG Systems
8 HippoRAG2 DM Linked 1.5 42.9 66.4 41.5 58.6 34.9 34.1 11.4 35.0 10.2
9 HippoRAG2 SGM Linked 1.5 47.769.646.9 59.4 39.3 41.8 17.636.1 15.3
10 Self-RAG DM Piled 0.5 42.1 61.8 42.8 46.4 33.1 30.9 14.0 34.1 10.9
11 Self-RAG SGM Piled 0.5 50.3 68.7 48.2 59.7 35.348.4 16.1 38.1 13.4
12 ATM-RAG DM Piled 0.5 42.0 61.8 41.3 50.6 31.0 38.9 14.1 34.1 11.4
13 ATM-RAG SGM Piled 0.5 51.068.7 48.6 60.3 32.4 48.2 16.3 38.1 13.7
Wang et al., 2021), operating over text-based memory items
produced by the Memory Ingestion pipeline. Alternative
embedding models and reranking setups are studied in Sec-
tion 5.4. All retrieval-based methods use a top- k=10 evi-
dence budget.
Models.All baselines use Qwen3-VL-2B-Instruct as the
memory processor (Bai et al., 2025) to ingest raw memory
data into unified memory M. This model is used con-
sistently for memory preprocessing and, when applicable,
memory organization. We adopt Qwen3-VL-2B-Instruct
because of its compact size, which makes it suitable for
on-device or edge deployment and helps mitigate privacy
concerns associated with personal memory data. Empiri-
cally, we observe little performance differences compared
to the larger 8B variant. For answer generation, we use
Qwen3-VL-8B-Instruct (Bai et al., 2025). At inference time,
both the retrieved memory items {mi}and the correspond-
ing raw visual inputs (images and videos) are provided to
the answerer. All video uses at most 8 frames to avoid ex-
hausting the maximum context-length. We use the same
prompt template and temperature for all systems to ensure
controlled comparison.
Metrics.We report results by question type (Number /
Recall-list / Open-ended questions, denoted as N/R/O), the
overall QS score, retrieval recall (R@10), and the joint met-
ric (Joint@10). By default, we use GPT-5-mini as the
LLM-based judge for open-ended answers. In Appendix E.1,
we compare different LLM-based judge models and observenearly identical performance across them. We report mem-
ory encoding time Tenc(hours), which includes Memory
Ingestion + embedding and retrieval index construction.
Linked methods additionally include graph construction.
No-Evidence and Oracle.We implement a lower-bound
baseline that answers directly from the question without ac-
cessing any memory evidence and an upper-bound (Oracle)
setting in which the answerer is provided with gold evidence.
This enables us to compare DM versus SGM Memory Rep-
resentation for answer generation under perfect recall.
ATM-RAG.We implemented a standard Multimodal
RAG pipeline similar to RA VQA (Lin et al., 2024) and
VISTA (Zhou et al., 2024). It uses either DM or SGM rep-
resentation, Piled Memory organization, and a single-pass
answer generator. (See Figure 2)
More Baselines.A-Mem (Xu et al., 2025),
Mem0 (Chhikara et al., 2025), and HippoRAG2 (Guti ´errez
et al., 2025) employ Linked Memory Organizations,
whereas Self-RAG (Asai et al., 2024) adopts a reflective
answer generation mechanism.
Implementation details are described in the Appendix D.
5.2. Main Findings
Table 3 present results on the ATM-Bench and the more
challenging ATM-Bench-Hard set.
7

Long-Term Personalized Referential Memory QA
SGM outperforms DM.Rows #2 and #3 show that SGM
consistently outperforms DM under the Oracle setup by 20%
improvement on ATM-Bench-Hard. Furthermore, results in
#8–#13 shows SGM systems consistently achieves higher
retrieval performance as well as higher downstream QS
scores for both linked and piled memory setups compared to
systems with DM. This holds across HippoRAG, Self-RAG,
and ATM-RAG systems.
Agentic answerer yields mixed gains.Rows #6 and #7
compare Mem0 variants that share the same Memory Inges-
tion but differ in the answer generation strategy. We find
agentic answerer improves performance on ATM-Bench,
but the gain does not transfer to ATM-Bench-Hard. These
results suggest that current agentic answering scheme is
not sufficient to address hard queries that requires long-
term reasoning and grounding. In comparing Self-RAG and
ATM-RAG, which differs primarily in having a reflective
versus single-pass answerer, we find their performance to
be comparable on ATM-Bench.
Linked vs. Piled Memory Organization.We modified
A-Mem to use Piled Memory organization and find perfor-
mance improves compared to its original Linked Memory
setting. This is achieved with substantially reduced en-
coding time (12.6h →1.6h). This suggests Piled Memory
organization can work well despite its simplicity.
5.3. Identifying Bottlenecks in Answer Generation
Table 4.Comparing different models under Oracle retriever. All
Qwen models are Instruct variants. All the reasoning model using
medium as the reasoning effort.
Answerer QS QS (Hard)
Qwen3-VL-2B 34.8 34.1
Qwen3-VL-8B 77.8 47.3
Gemini 2.5 Flash 76.9 55.6
Gemini 2.5 Pro 78.6 64.3
Claude Sonnet 4.5 83.3 61.9
Claude Opus 4.586.062.7
GPT-5 85.374.7
Table 4 compares performance of representative open- and
closed-source multimodal LLMs under the oracle setup. On
ATM-Bench, several frontier models achieve high QS, with
Claude Opus 4.5 reaching 86.0 and GPT-5 achieving 85.3.
However, performance drops substantially on ATM-Bench-
Hard: GPT-5, the best-performing model, attains 74.7, while
the runner-up Gemini 2.5 Pro attains 64.3. These results
indicate that even with the ground-truth memory items, long-
term personal Memory QA remains challenging. A likely
contributing factor is the need to aggregate and reconcile
a larger number of memory items. We provide qualitative
error analysis in Appendix F.5.4. Comparing different Retrievers and Rerankers
Table 5.Retrieval and rerank comparison for ATM-RAG. For
reranking, we retrieve top-20 candidates and rerank them to top-
10.
Embedding Reranker QS R@10 Joint@10
MiniLM-L6 – 51.0 68.7 48.6
MiniLM-L6 0.6B 53.3 69.9 49.0
Qwen3-0.6B – 53.2 71.5 51.5
Qwen3-0.6B 0.6B 52.0 69.6 48.5
Qwen3-0.6B 4B55.873.6 51.8
Qwen3-4B –54.5 75.2 52.8
Qwen3-VL-2B – 31.3 40.1 28.5
We study retrieval configurations using ATM-RAG to un-
derstand which retrieval setup best supports multi-source,
multimodal personal Memory QA.
Retriever and Reranker Variants.In addition to all-
MiniLM-L6-v2, we evaluate Qwen3-Embedding (Zhang
et al., 2025) and Qwen3-VL-Embedding models (Li et al.,
2026). When reranking is enabled, we compare 0.6B and
4B Qwen3-Reranker (Zhang et al., 2025). We report the
indexing cost in Appendix E.3
Results and Analysis.Table 5 reports retrieval and end-
to-end performance. We observe that scaling up the retriever
embedding model generally improves retrieval recall and
downstream QA performance. Surprisingly, we find that
using Qwen3-VL-Embedding-2B as the embedding model
results in worse performance compared to its text-based
counterparts. A possible explanation is that a large num-
ber of visual tokens (e.g., ∼2,500 tokens per image) are
used to encode the high-resolution (2 MegaPixels) image
in the benchmark. This likely dilutes the critical metadata
signals such as timestamps and location information. Ad-
ditional ablations on the retrieval depth ( k) are provided in
Appendix E.2.
6. Conclusion
We introduced ATM-Bench, the first benchmark for per-
sonalized Memory QA over long-term, multimodal, and
multi-source personal memory data. We proposed Schema-
Guided Memory to unify heterogeneous memory sources.
Evaluations of state-of-the-art memory systems reveal sub-
stantial performance gaps, with accuracy remaining below
20% on the challenging ATM-Bench-Hard subset. These
results highlight the limitations of current approaches and
motivate future research on robust and scalable personalized
memory systems.
8

Long-Term Personalized Referential Memory QA
Impact Statement
This paper introduces ATM-BENCH, a research benchmark
designed to advance machine learning methods for personal
memory retrieval and reasoning. The primary anticipated
impact of this work is to support the development and eval-
uation of privacy-aware, multimodal memory systems in a
controlled and reproducible setting.
Ethical considerations are a core priority of this work. We
conducted a comprehensive risk and ethical analysis, de-
tailed in Appendix A, covering data provenance, privacy
preservation, anonymization, content safety, and intended
use. All data included in ATM-BENCHwas collected with
explicit informed consent and processed through a multi-
stage anonymization pipeline combining automated filtering
and human verification to ensure the removal of personally
identifiable information and sensitive content. The released
benchmark contains no real PII and avoids encoding or
inferring protected attributes.
We further restrict the benchmark and the methodology to
non-commercial research use and explicitly prohibit appli-
cations related to surveillance, profiling, or consequential
decision-making about individuals. As such, we view ATM-
BENCHas a methodological testbed rather than a deployable
personal assistant system.
Overall, we believe this work contributes positively to the
field by enabling rigorous research on long-term memory
modeling while demonstrating that such research can be con-
ducted under strong ethical, privacy, and safety constraints.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H.
Self-rag: Learning to retrieve, generate, and critique
through self -reflection. InProceedings of the 12th
International Conference on Learning Representations
(ICLR 2024). International Conference on Learning Rep-
resentations, 2024. URL https://openreview.
net/forum?id=hSyW5go0v8.
Bai, S., Cai, Y ., Chen, R., Chen, K., Chen, X., Cheng, Z.,
Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z.,
Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li,
Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu,
J., Liu, C., Liu, Y ., Liu, D., Liu, S., Lu, D., Luo, R.,
Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S.,
Sun, Y ., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P.,
Wang, Q., Wang, Y ., Xie, T., Xu, Y ., Xu, H., Xu, J.,
Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang,
F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou,
J., Zhou, F., Zhou, J., Zhu, Y ., and Zhu, K. Qwen3-
vl technical report.CoRR, abs/2511.21631, 2025. doi:
10.48550/ARXIV .2511.21631. URL https://doi.org/10.48550/arXiv.2511.21631.
Chen, Y ., Hu, H., Luan, Y ., Sun, H., Changpinyo, S., Ritter,
A., and Chang, M.-W. Can pre-trained vision and lan-
guage models answer visual information-seeking ques-
tions? In Bouamor, H., Pino, J., and Bali, K. (eds.),
Proceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 14948–14968,
Singapore, December 2023. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2023.emnlp-main.
925. URL https://aclanthology.org/2023.
emnlp-main.925/.
Chhikara, P., Khant, D., Aryan, S., Singh, T., and Ya-
dav, D. Mem0: Building production -ready AI agents
with scalable long -term memory, 2025. URL https:
//arxiv.org/abs/2504.19413.
Du, Y ., Wang, H., Zhao, Z., Liang, B., Wang, B., Zhong, W.,
Wang, Z., and Wong, K. PerLTQA: A personal long -term
memory dataset for memory classification, retrieval, and
synthesis in question answering, 2024. URL https:
//arxiv.org/abs/2402.16288.
Guti´errez, B. J., Shu, Y ., Qi, W., Zhou, S., and Su, Y .
From rag to memory: Non-parametric continual learn-
ing for large language models. 2025. URL https:
//openreview.net/forum?id=LWH8yn4HS2.
Harvey, M., Langheinrich, M., and Ward, G. Remembering
through lifelogging: A survey of human memory augmen-
tation.Pervasive and Mobile Computing, 27:14–26, April
2016. ISSN 1574-1192. doi: 10.1016/j.pmcj.2015.12.
002. URL https://doi.org/10.1016/j.pmcj.
2015.12.002.
Hosseini, P., Castro, I., Ghinassi, I., and Purver, M. Efficient
solutions for an intriguing failure of LLMs: Long context
window does not mean LLMs can analyze long sequences
flawlessly. In Rambow, O., Wanner, L., Apidianaki, M.,
Al-Khalifa, H., Eugenio, B. D., and Schockaert, S. (eds.),
Proceedings of the 31st International Conference on Com-
putational Linguistics, pp. 1880–1891, Abu Dhabi, UAE,
January 2025. Association for Computational Linguis-
tics. URL https://aclanthology.org/2025.
coling-main.128/.
Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni,
F., Schick, T., Dwivedi -Yu, J., Joulin, A., Riedel, S.,
and Grave, E. Atlas: Few -shot learning with retrieval
augmented language models.arXiv preprint, 2022. doi:
10.48550/arXiv.2208.03299. URL https://arxiv.
org/abs/2208.03299.
Jain, C., Wu, Y ., Zeng, Y ., Liu, J., Dai, S., Shao, Z., Wu, Q.,
and Wang, H. Simpledoc: Multi -modal document under-
standing with dual -cue page retrieval and iterative refine-
9

Long-Term Personalized Referential Memory QA
ment. InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pp. 28410–
28427, Suzhou, China, 2025. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2025.emnlp-main.
1443. URL https://aclanthology.org/2025.
emnlp-main.1443/.
Jiang, H., Zhang, X., Garg, S., Arora, R., Kuo, S.-Z.,
Xu, J., Colak, A., and Dong, X. L. Memory-QA: An-
swering recall questions based on multimodal memories.
InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pp. 24244–
24266, Suzhou, China, 2025. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2025.emnlp-main.
1234. URL https://aclanthology.org/2025.
emnlp-main.1234/.
Jin, M., Luo, W., Cheng, S., Wang, X., Hua, W., Tang,
R., Wang, W. Y ., and Zhang, Y . Disentangling mem-
ory and reasoning ability in large language models. In
Proceedings of the 63rd Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long
Papers), pp. 1681–1701, Vienna, Austria, 2025. Associ-
ation for Computational Linguistics. doi: 10.18653/v1/
2025.acl-long.84. URL https://aclanthology.
org/2025.acl-long.84/.
Kang, J., Ji, M., Zhao, Z., and Bai, T. Memory OS of AI
agent. InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pp. 25961–
25970, Suzhou, China, 2025. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2025.emnlp-main.
1318. URL https://aclanthology.org/2025.
emnlp-main.1318/.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. InPro-
ceedings of the 2020 Conference on Empirical Meth-
ods in Natural Language Processing (EMNLP), pp.
6769–6781, Online, November 2020. Association for
Computational Linguistics. doi: 10.18653/v1/2020.
emnlp-main.550. URL https://aclanthology.
org/2020.emnlp-main.550.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin,
V ., Goyal, N., K ¨uttler, H., Lewis, M., Yih,
W., Rockt”aschel, T., Riedel, S., and Kiela, D.
Retrieval -augmented generation for knowledge -intensive
NLP tasks. InAdvances in Neural Information
Processing Systems 33 (NeurIPS 2020), pp. 9459–
9474, Vancouver, BC, Canada, 2020. Curran As-
sociates, Inc. URL https://proceedings.
neurips.cc/paper/2020/hash/
6b493230205f780e1bc26945df7481e5-Abstract.
html.Li, J. N., Zhang, Z. J., and Ma, J. Omniquery: Contex-
tually augmenting captured multimodal memory to en-
able personal question answering. InProceedings of the
2025 CHI Conference on Human Factors in Computing
Systems, CHI ’25, New York, NY , USA, 2025. Associa-
tion for Computing Machinery. ISBN 9798400713941.
doi: 10.1145/3706598.3713448. URL https://doi.
org/10.1145/3706598.3713448.
Li, M., Zhang, Y ., Long, D., Chen, K., Song, S., Bai, S.,
Yang, Z., Xie, P., Yang, A., Liu, D., Zhou, J., and Lin, J.
Qwen3-vl-embedding and qwen3-vl-reranker: A unified
framework for state-of-the-art multimodal retrieval and
ranking. (arXiv:2601.04720), January 2026. doi: 10.
48550/arXiv.2601.04720. URL http://arxiv.org/
abs/2601.04720. arXiv:2601.04720 [cs].
Li, Y ., Wen, H., Wang, W., Li, X., Yuan, Y ., Liu, G., Liu, J.,
Xu, W., Wang, X., Sun, Y ., Kong, R., Wang, Y ., Geng, H.,
Luan, J., Jin, X., Ye, Z., Xiong, G., Zhang, F., Li, X., Xu,
M., Li, Z., Li, P., Liu, Y ., Zhang, Y .-Q., and Liu, Y . Per-
sonal llm agents: Insights and survey about the capability,
efficiency and security.arXiv preprint arXiv:2401.05459,
2024. doi: 10.48550/arXiv.2401.05459. URL http:
//arxiv.org/abs/2401.05459.
Lin, W., Mei, J., Chen, J., and Byrne, B. Preflmr:
Scaling up fine-grained late-interaction multi-modal re-
trievers. In Ku, L.-W., Martins, A., and Srikumar,
V . (eds.),Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 5294–5316, Bangkok, Thailand,
August 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.acl-long.289. URL https:
//aclanthology.org/2024.acl-long.289/.
Liu, J., Zhu, D., Bai, Z., He, Y ., Liao, H., Que, H., Wang,
Z., Zhang, C., Zhang, G., Zhang, J., Zhang, Y ., Chen, Z.,
Guo, H., Li, S., Liu, Z., Shan, Y ., Song, Y ., Tian, J., Wu,
W., Zhou, Z., Zhu, R., Feng, J., Gao, Y ., He, S., Li, Z.,
Liu, T., Meng, F., Su, W., Tan, Y ., Wang, Z., Yang, J., Ye,
W., Zheng, B., Zhou, W., Huang, W., Li, S., and Zhang,
Z. A comprehensive survey on long context language
modeling. (arXiv:2503.17407), November 2025. doi: 10.
48550/arXiv.2503.17407. URL http://arxiv.org/
abs/2503.17407. arXiv:2503.17407 [cs].
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts.Transactions of
the Association for Computational Linguistics, 12:157–
173, 2024. doi: 10.1162/tacl a00638. URL https:
//aclanthology.org/2024.tacl-1.9/.
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbi-
eri, F., and Fang, Y . Evaluating very long -term conver-
sational memory of LLM agents. InProceedings of the
10

Long-Term Personalized Referential Memory QA
62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pp. 13851–
13870, Bangkok, Thailand, 2024. Association for Com-
putational Linguistics. doi: 10.18653/v1/2024.acl-long.
747. URL https://aclanthology.org/2024.
acl-long.747/.
MemoDB. Memobase, 2024. URL https://github.
com/memodb-io/memobase.
Mensink, T., Uijlings, J., Castrejon, L., Goel, A., Cadar,
F., Zhou, H., Sha, F., Araujo, A., and Ferrari, V . En-
cyclopedic vqa: Visual questions about detailed proper-
ties of fine-grained categories. In2023 IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pp.
3090–3101, Paris, France, October 2023. IEEE. ISBN
979-8-3503-0718-4. doi: 10.1109/ICCV51070.2023.
00289. URL https://ieeexplore.ieee.org/
document/10376881/.
Packer, C., Wooders, S., Lin, K., Fang, V ., Patil, S. G.,
Stoica, I., and Gonzalez, J. E. MemGPT: Towards LLMs
as operating systems, 2024. URL https://arxiv.
org/abs/2310.08560.
Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N., and
Lewis, M. Measuring and narrowing the compositionality
gap in language models. InFindings of the Association
for Computational Linguistics: EMNLP 2023, pp. 5687–
5711, Singapore, 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.findings-emnlp.
378. URL https://aclanthology.org/2023.
findings-emnlp.378/.
Reimers, N. and Gurevych, I. Sentence-bert: Sen-
tence embeddings using siamese bert-networks.CoRR,
abs/1908.10084, 2019. URL http://arxiv.org/
abs/1908.10084.
Shao, Y ., Jiang, Y ., Kanell, T. A., Xu, P., Khattab, O., and
Lam, M. S. Assisting in writing wikipedia -like articles
from scratch with large language models. InProceedings
of the 2024 Annual Conference of the North American
Chapter of the Association for Computational Linguistics
(NAACL 2024). Association for Computational Linguis-
tics, 2024. URL https://aclanthology.org/
2024.naacl-main.106/.
Tan, H., Zhang, Z., Ma, C., Chen, X., Dai, Q., and Dong,
Z. MemBench: Towards more comprehensive evaluation
on the memory of LLM-based agents. InFindings of the
Association for Computational Linguistics: ACL 2025,
pp. 19336–19352, Vienna, Austria, 2025. Association
for Computational Linguistics. doi: 10.18653/v1/2025.
findings-acl.989. URL https://aclanthology.
org/2025.findings-acl.989/.Wang, W., Bao, H., Huang, S., Dong, L., and Wei,
F. Minilmv2: Multi-head self-attention relation dis-
tillation for compressing pretrained transformers. In
Zong, C., Xia, F., Li, W., and Navigli, R. (eds.),
Findings of the Association for Computational Linguis-
tics: ACL/IJCNLP 2021, Online Event, August 1-6,
2021, volume ACL-IJCNLP 2021 ofFindings of ACL,
pp. 2140–2151. Association for Computational Linguis-
tics, 2021. doi: 10.18653/V1/2021.FINDINGS-ACL.
188. URL https://doi.org/10.18653/v1/
2021.findings-acl.188.
Wang, Y ., Krotov, D., Hu, Y ., Gao, Y ., Zhou, W., McAuley,
J., Gutfreund, D., Feris, R., and He, Z. M+: Ex-
tending MemoryLLM with scalable long -term mem-
ory. InProceedings of the 42nd International Confer-
ence on Machine Learning (ICML 2025), 2025. doi:
10.48550/arXiv.2502.00592. URL https://arxiv.
org/abs/2502.00592.
Wu, D., Wang, H., Yu, W., Zhang, Y ., Chang, K.-W., and
Yu, D. Longmemeval: Benchmarking chat assistants
on long-term interactive memory. InProceedings of
the 13th International Conference on Learning Repre-
sentations (ICLR 2025), 2025. doi: 10.48550/arXiv.2410.
10813. URL https://openreview.net/forum?
id=pZiyCaVuti.
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang,
Y . A-MEM: Agentic memory for LLM agents. InAd-
vances in Neural Information Processing Systems 38
(NeurIPS 2025), 2025. doi: 10.48550/arXiv.2502.12110.
URLhttps://arxiv.org/abs/2502.12110.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. Hotpotqa: A
dataset for diverse, explainable multi-hop question an-
swering. In Riloff, E., Chiang, D., Hockenmaier, J.,
and Tsujii, J. (eds.),Proceedings of the 2018 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, Brussels, Belgium, October 31 - November 4,
2018, pp. 2369–2380. Association for Computational
Linguistics, 2018. doi: 10.18653/V1/D18-1259. URL
https://doi.org/10.18653/v1/d18-1259.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang,
B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F.,
and Zhou, J. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
CoRR, abs/2506.05176, 2025. doi: 10.48550/ARXIV .
2506.05176. URLhttps://doi.org/10.48550/
arXiv.2506.05176.
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y . Memory-
bank: Enhancing large language models with long-term
11

Long-Term Personalized Referential Memory QA
memory.Proceedings of the AAAI Conference on Ar-
tificial Intelligence, 38(17):19724–19731, March 2024.
ISSN 2374-3468. doi: 10.1609/aaai.v38i17.29946.
Zhou, J., Liu, Z., Xiao, S., Zhao, B., and Xiong, Y .
VISTA: visualized text embedding for universal multi-
modal retrieval. In Ku, L., Martins, A., and Sriku-
mar, V . (eds.),Proceedings of the 62nd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2024, Bangkok, Thai-
land, August 11-16, 2024, pp. 3185–3200. Association
for Computational Linguistics, 2024. doi: 10.18653/
V1/2024.ACL-LONG.175. URL https://doi.org/
10.18653/v1/2024.acl-long.175.
12

Long-Term Personalized Referential Memory QA
A. Risk and Ethical Assessment
The development ofATM-Benchadheres to strict ethical guidelines regarding data privacy, user consent, and content safety.
We have implemented a multi-stage protocol and conducted a comprehensive internal risk and ethical assessment to ensure
the dataset serves the research community without compromising the anonymity or safety of its contributors. Here, we
provide the full detail of the privacy and Anonymization processing pipelines, the statement on data privacy, content safety,
and intended use.
A.1. Privacy and Anonymization Pipeline
Protecting the identity of data contributors is our priority. To ensure the release of a safe, privacy-preserving benchmark, we
designed a multi-layer anonymization pipeline that combines automated AI filtering with human verification.
Image and VideoVisual data underwent a three-stage filtering process:
1.VLM Risk Assessment:We first employed two Vision-Language Models (Qwen2.5-VL-7B and GPT-4o-mini) to
screen all raw visual data for sensitive content. We utilized the prompt below to flag any potential risks; ifeithermodel
flagged an image as ‘sensitive’ or ‘private’, it was automatically discarded.
VLM Safety Prompt
Analyze this image for any sensitive content that should be flagged:
• Personal information (IDs, documents, credit cards, etc.)
• Private/intimate moments
• Children in potentially sensitive contexts
• Any content that should be treated with extra privacy
Return a single word: ‘safe’, ‘sensitive’, or ‘private’.
2.Automated Face Blurring:On the remaining ‘safe’ images, we applied a standard face detection and blurring
algorithm to obscure human faces.
3.Manual Verification & Fine-Grained Redaction:A human annotation team reviewed every processed image and
video frame. This step involved:
• Verifying that automated face blurring was successful.
• Manually applying blur to any missed faces.
•Redacting private attributes like text and bar-codes. (e.g., names on conference badges, ticket numbers, receipts,
or background computer screens).
Email ProcessingTextual data required a different approach to remove linguistic fingerprints and metadata:
1.Metadata Stripping:We systematically removed all header information, including sender/receiver names, signature,
and server paths.
2.LLM Rewriting:To prevent re-identification through writing style, we prompted Qwen2.5-7B to paraphrase the body
content while preserving the core semantic information (e.g., event details, purchase items).
13

Long-Term Personalized Referential Memory QA
Email Rewriting & Sanitization Prompt
Task:Generate a privacy-preserving version of an email.
Your output will be used in a public dataset, so strict adherence to privacy rules is paramount. Any leakage of
Personally Identifiable Information (PII) or sensitive data is unacceptable.
Email Details:
Date:{timestamp}From:{sender}
Email Content: —
{email body}—
Please rewrite the email based on the above content. The rewrite should be within 5 sentences long and focus
on:
• Remove Real email addresses or real names. Use general but informative references.
•Rewrite the subject if it contains privacy leaking info. Rewrite the subject to better summarize the content.
• Focus on the semantics of the message, removing any sensitive or irrelevant detail.
• Output only the summary text — no bullet points, labels, or explanations.
•If contains shopping list, or detailed invoice, list all the relevant products/services. Note that public
addresses (store/hotel location/name) are not sensitive and should be included.
STRICT PRIV ACY AND REDACTION RULES:
• Absolutely no PII or sensitive data is permitted.
•You MUST NOT mention: Real names (use [PERSON], [SENDER]), Phone numbers, Private residential
addresses, ID numbers (passport, license, SSN), Real email addresses, IP addresses, Full URLs, Financial
account details, Transactional/Tracking numbers.
Exceptions:Public addresses of known establishments (restaurants, hotels) and Company/Product names are
allowed.
Output Format:Date: [e.g., ’2023-01-15’]
From: [Privacy-preserving sender]
Subject: [Summarized Subject]
Content: [The 3-5 sentence detailed abstract...]
3.Synthetic PII Injection:We applied a “PII Replacement” protocol to systematically swap specific entities with
contextually appropriate synthetic placeholders. For example, real tracking numbers were replaced with consistent
dummy formats (e.g., 1Z999... ) and private addresses were mapped to fictitious but realistic locations. This step
preserves the structural complexity of the data, ensuring we can still benchmark agents on specific retrieval tasks (e.g.,
“What is the tracking number for my package?”) without exposing sensitive user information.
4.Final Human Audit:Annotators performed a final comprehensive review of all rewritten emails to ensure no residual
identifiers or sensitive context remained.
The total human verification effort for this stage amounted to approximately 150 hours.
A.2. Data Provenance and Anonymization Statement
Informed Consent.All personal memory data included in the raw collection—spanning emails, photos, and videos—was
obtained with explicit, informed consent from the contributors. Participants provided written authorization for their data to
be processed, anonymized, and released specifically for academic research purposes.
PII and De-Identification.The released benchmark containsnoreal Personally Identifiable Information (PII). While the
data reflects real-world events, we applied a comprehensive sanitization pipeline as mentioned in Appendix A.1.
14

Long-Term Personalized Referential Memory QA
A.3. Content Safety and Protected Attributes
We explicitly filtered the dataset to exclude content that is offensive, harmful, or inappropriate.
•Harmful Content:The data selection and generation pipeline is constrained to strictly avoid hate speech, harassment,
sexual content, and depictions of self-harm or violence.
•Protected Attributes:For the released benchmark, protected attributes such as sexual orientation, detailed religious
belief, political affiliation, and specific health diagnoses arenotencoded or inferred. The generation templates avoid
making these attributes central to the tasks.
•Demographic Control:We include only a limited set ofcoarsedemographic attributes (e.g., broad region) at the
persona level solely to support controlled diversity analysis and debiasing. No characteristics of the real subjects (e.g.,
age, gender, identity) were self-reported or inferred during analysis.
A.4. Intended Use and Limitations
ATM-BENCHis released strictly fornon-commercial research purposesto support the development of personal memory
retrieval systems under controlled, privacy-preserving conditions.
Licensing.The dataset is distributed under theCreative Commons Attribution-NonCommercial 4.0 International
(CC-BY-NC 4.0)license. The accompanying code is released under the MIT License.
Prohibited Uses.We explicitly prohibit the use of our benchmark or code:
• For surveillance, monitoring, or profiling of individuals.
•To build systems that infer sensitive attributes or make consequential decisions (e.g., hiring, credit, insurance, law
enforcement) about real people.
•To justify intrusive data collection practices beyond what is ethically acceptable and legally compliant in the relevant
jurisdiction.
We view our contribution as providing a concrete, privacy-preserving testbed for methodology development, not as a
ready-to-deploy system for real-world personal assistance.
B. Annotation
B.1. Failure Modes of Agentic Auto-Annotation
After data collection, the raw memory corpus contained approximately 13,000 items spanning images, videos, and emails.
We experimented with multiple agentic auto-annotation pipelines using state-of-the-art large language and multimodal
models (e.g., Opus-4, Gemini-2.5-Pro) to automatically generate question–answer–evidence (QAE) triples.
While these models were generally capable of identifying temporally and spatially relevant memory evidence, human
evaluation revealed a systematic misalignment between auto-generated questions and realistic human memory queries.
Specifically, many generated questions emphasized analytical properties of the data rather than human recall behavior.
Examples include questions such as“How many days after booking did the stay start?”or“What wildlife did I encounter
during my hiking trip on May 22?”. Although such questions test a model’s ability to retrieve correct memories, they are
unlikely to reflect how users naturally remember past experiences.
In practice, human memory recall is often triggered by salient entities or events rather than precise temporal offsets. For
instance, a user is more likely to remember seeing a deer during a hike than to recall the exact date or time interval associated
with the photo. A more realistic human query would therefore take the form: “I remember seeing and taking a photo of a
deer during a recent hiking trip—when and where did that happen?” This type of question emphasizes entity-based recall
and contextual grounding, which were rarely produced by agentic auto-annotation.
15

Long-Term Personalized Referential Memory QA
Figure 3.The UI of the annotation tool
Based on these observations, we determined that fully automated annotation pipelines—even when using state-of-the-art
models—fail to capture the distribution of human-likely personal memory questions. Instead, ATM-Bench adopts a human-
centered annotation strategy informed by prior HCI studies on personal memory querying behavior (Li et al., 2025). All five
human annotators reviewed and discussed the annotation principles derived from this study prior to annotation, ensuring that
the resulting questions reflect realistic, experience-driven user queries.
B.2. Human Annotation Protocol
The ATM-Bench annotation process is carried out by five trained human annotators and requires approximately 200 total
annotation hours (about 40 hours per annotator). Annotation is performed over a corpus of personal memory items spanning
images, videos, and emails.
Annotation WorkflowAnnotation proceeds in three stages. First, question–answer–evidence (QAE) pairs are annotated
separately for email memories and for visual memories (images and videos). In the second stage, annotated items are
temporally sorted and merged, enabling the construction of questions that require reasoning across multiple memory sources.
Annotators are presented with collections of memory items grouped within a temporal window of approximately one year
and are instructed to review all relevant memories within that period prior to authoring questions.
Annotation ToolingTo facilitate cross-modal evidence gathering, annotators are provided with an interactive search
interface that supports querying OCR text, named entities, semantic tags, and caption keywords across all memory sources.
This tooling enables efficient navigation of the memory corpus and supports the construction of questions grounded in
evidence spanning multiple modalities.
Quality Control and ValidationFollowing initial annotation, each QAE pair undergoes a second-round validation by a
different annotator. This validation step verifies that (i) the question is grounded in the provided memory evidence, (ii) the
evidence set is complete and sufficient to answer the question, and (iii) no irrelevant or temporally inconsistent memory
items are included. Disagreements are resolved through discussion, and only validated QAE pairs are retained in the final
benchmark.
16

Long-Term Personalized Referential Memory QA
Annotation GuidelinesAnnotators follow a shared set of annotation guidelines informed by prior HCI studies on realistic
personal memory queries. These guidelines emphasize human-likely question formulation, entity- and event-driven recall,
and appropriate temporal and spatial grounding.
B.3. Personal Memory QA Annotation Guidelines
Task Definition.Annotators constructPersonal Memory Question–Answer–Evidence (QAE)triplets from a user’s photo
gallery. Each gallery corresponds to a coherent time period or event. The objective is to model realistic human memory
recall: questions that a person might naturally ask later when attempting to retrieve their own past experiences.
Valid Personal Memory QA.A valid Personal Memory QA must satisfy all of the following criteria:
•First-person perspective.Questions must be written as if asked by the photo owner (e.g., “Where did I . . . ”, “When
did I . . . ”, “I remember . . . ”).
•Partial or fuzzy recall.Questions should reflect incomplete memory and must not explicitly reveal the answer.
•Evidence grounding.Both the question and the answer must be fully supported by the selected image(s) and their
associated metadata (e.g., timestamp, location).
•Retrieval necessity.Answering the question should require retrieving the correct memory item(s), rather than relying
on surface-level visual description or general world knowledge.
What to Annotate.Only annotatememorable or distinctivemoments, such as travel experiences, special events, or
unusual encounters. It is acceptable—and expected—to skip most images. As a rough guideline, annotating approximately
100 QA pairs from 800 images is considered a reasonable density.
What Not to Annotate.Annotators should avoid creating QAs for:
• Generic or routine daily-life scenes (e.g., familiar streets, queues, repetitive meals).
• Highly ambiguous situations that could correspond to many different memories.
• Questions whose answers are trivial visual descriptions and do not require memory retrieval.
Question Writing Principles.Questions should be specific enough to identify a single memory, while avoiding explicit
disclosure of the answer. Annotators are encouraged to use distinctive cues such as events, activities, social context, or
unusual objects. Questions should not be overly vague nor overly explicit.
For temporal queries, coarse-grained references are preferred:
• Ask at the day or event level rather than precise timestamps.
• Avoid exact clock times.
Answer Writing Principles.Answers must be short, factual, and precise. Valid answers may include:
• a date (e.g., “14 December 2023”),
• a place name (e.g., restaurant, landmark, or city),
• or a combination of both.
Annotators should not speculate beyond what is directly supported by the selected evidence.
Evidence Selection.Onlyprimary evidence imagesthat are strictly necessary to answer the question should be selected.
In most cases, a single image is sufficient; multiple images may be used only if jointly required. Redundant or secondary
images should not be included.
17

Long-Term Personalized Referential Memory QA
<1d 1d 1w 2w1mo 3mo 3mo+
(c)0102030405060Count250500750
1d 3d 1w 2w 2w+
(b)02468101214250500750
1 2 3 4 4+
(a)0255075100125150
Figure 4.(a) Evidence count. (b) Maximum time span between evidence items for a single QA instance. (c) Recall distance.
Quality Control.Before finalizing a QAE triplet, annotators should verify the following:
1. Is this something a person might genuinely forget and later want to recall?
2. Does the question uniquely identify one specific memory?
3. Is the answer fully supported by the selected evidence?
4. Does answering require correct memory retrieval rather than superficial description?
If any of these conditions are not met, the QA should be revised or discarded.
C. Dataset Composition, Statistics and Analysis
C.1. Statistics
We report three per-question statistics computed from the evidence items associated with each question, as illustrated in
Figure 4.
Evidence Count (Figure 4 (a))This corresponds to the number of unique evidence IDs linked to a given question.
Maximum time span between evidence items (Figure 4 (b))For the time-based statistics, we extract an optional anchor
date from the question text when it contains a “Today is . . . ” pattern (formatted as YYYY-MM-DD). Themaximum time
span between evidence itemsfor a single QA instance is defined as the time difference (in days) between the earliest and
latest evidence timestamps. This value is undefined when fewer than two evidence timestamps are available.
Recall Distance (Figure 4 (c))Recall distance is defined as the time difference (in days) between the anchor date and the
most recent evidence timestamp. This statistic is undefined if the anchor date is missing or if no evidence timestamps are
available.
Question type CountsIn table 6, we report the question type for both the ATM-Bench and ATM-Bench-Hard. The
abstention are counted in open end, which in total counts 25, account for about 0.3%.
Semantic TaxonomyEach question is annotated with two orthogonal semantic labels:Topic, indicating the domain of the
underlying memory, andIntent, specifying the reasoning operation required to answer the question. Evidence modality
(email, image, or video) is recorded separately and is not encoded in these semantic labels.
Topic taxonomy.We define six topic categories:Work + Learning & Knowledge(professional work, research, coursework,
meetings, deadlines, administration, and technical troubleshooting);Personal Information (systematically generated to
remove PII)(identity- and record-oriented items such as IDs, healthcare, memberships, and personal profiles);Travel
18

Long-Term Personalized Referential Memory QA
Table 6.Qtype counts.
ATM-Bench
Qtype Count
open end 514
number 360
listrecall 139ATM-Bench-Hard
Qtype Count
number 6
listrecall 12
open end 7
Table 7.Semantic Classification.
Topic distribution.
Topic Count
Work + Learning & Knowledge 208
Activities 374
Household 30
Travel 191
Everyday Life Logging 157
Personal Information (privacy preserved and sanitized) 78Intent distribution.
Intent Count
Identification 67
Geolocation 198
Aggregation 39
Recall 356
Temporal 378
(multi-day or long-distance trips, including flights, accommodation, and itineraries);Activities(local outings, appointments,
dining, shopping, exercise, and events);Household(home-related operations such as rent, utilities, maintenance, deliveries,
and shared living logistics); andEveryday Life Logging(routine diary-like entries or minor episodic memories not covered
by other categories). The topic distribution over the combined dataset is shown in Table 7.
Intent taxonomy.We define five intent categories:Temporal(reasoning about time, order, duration, or frequency);Geolo-
cation(inferring or reporting locations or routes);Recall(retrieving specific content rather than metadata);Identification
(naming or disambiguating entities); andAggregation(computing summary statistics such as counts, totals, averages, or
trends). The intent distribution is reported in Table 7.
Memory Topic.Table 8 reports the distribution of memory items across semantic topics and evidence modalities, excluding
entries labeled as Unknown. Emails dominateWork + Learning & Knowledge, reflecting the prevalence of professional
communication and record-keeping in textual form. In contrast,Activitiesare primarily captured through images and
videos, consistent with real-world photo and video logging of daily events.Travelmemories are more evenly distributed
across modalities, as itineraries, confirmations, and receipts appear in emails, while locations and experiences are often
documented visually.Everyday Life Loggingspans all three modalities, indicating heterogeneous capture patterns for routine
events.HouseholdandPersonal Informationmemories are overwhelmingly email-centric, reflecting their administrative and
record-oriented nature. Overall, the table highlights strong modality–topic correlations that motivate multimodal memory
modeling.
Table 8.Memory topic distribution by modality.
Topic Email Image Video Combined
Work + Learning & Knowledge 3,826 396 3 4,225
Activities 1,341 2,491 339 4,171
Travel 421 434 78 933
Everyday Life Logging 216 340 105 661
Household 481 70 6 557
Personal Information (privacy preserved and sanitized) 455 21 2 478
Video Statistics.Table 9 reports the video length statistics.
19

Long-Term Personalized Referential Memory QA
Table 9.Video duration summary statistics (seconds). Values are mean±std with min/max and p95.
Count Mean±Std Min Max P95
533 13.0±20.0 0.4 207.5 35.7
180
 150
 120
 90
 60
 30
 0 30 60 90 120 150 180
Longitude60
30
0306090LatitudeGeographic Coverage of Image/Video Memories
Figure 5.Geo coverage.
C.2. Geo-Coverage
We cover a wide range of geographic regions in ATM-Bench, with image and video memories spanning countries across
North America, Europe, Asia, and Oceania. As shown in Figure 5, the highlighted countries indicate locations represented
in our benchmark. This broad geographic coverage enables evaluation of models under diverse visual environments, cultural
contexts, and real-world settings.
D. Implementation Details
D.1. Baseline Systems
A-Mem.An agentic memory system that writes structured memory notes and optionally organizes them using explicit
links constructed from keywords and tags. We evaluate both a piled memory variant and a linked variant that supports
link-based expansion, following the original implementation2.
Mem0 (Single-pass/Agentic).A memory layer that extracts and stores salient information persistently; we evaluate a
single-pass answerer variant and an agentic variant using the official api3.
HippoRAG2.A retrieval-augmented generation system that constructs a graph-based memory structure and expands
retrieval via graph traversal to identify related evidence. We follow the original implementation for the DM (D) representation.
2https://github.com/agiresearch/A-mem
3https://github.com/mem0ai/mem0
20

Long-Term Personalized Referential Memory QA
Table 10.Oracle retriever results with dual judges. QS and QS (Hard) are scored by GPT-5-mini; QS (GLM) and QS (Hard, GLM) are
scored by GLM-4.7. All Qwen models are Instruct variants. All reasoning models use medium reasoning effort.
Answerer QS (GPT) QS (Hard, GPT) QS (GLM) QS (Hard, GLM)
Qwen3-VL-8B (No evidence) 0.2 0 0.2 0
Qwen3-VL-2B 34.8 34.1 34.9 34.3
Qwen3-VL-8B 77.8 47.3 78.4 47.3
Gemini 2.5 Flash 76.9 55.6 77.0 56.6
Gemini 2.5 Pro 78.6 64.3 79.5 64.3
Claude Sonnet 4.5 83.3 61.9 84.2 62.7
Claude Opus 4.586.062.787.064.7
GPT-5 85.374.786.6 74.7
In addition, we adapt HippoRAG2 to operate over our proposed schema-driven (S) memory representation to study the
effect of structured memory within a graph-based retrieval framework. We adopt the original implementation4
Self-RAG.(Asai et al., 2024) We implement a self-reflective RAG5baseline that can validates evidence sufficiency
before generating the final answer. We evaluate Self -RAG under both descriptive (D) and schema -driven (S) memory
representations.
D.2. Environment
Environment. PyTorch 2.7.0 ,CUDA 12.8 , Huggingface Transformer 4.57.0 andPython 3.10.12
were used for implementing the experiments. Inference were conducted withvllm 0.13.0
Implementation environment.We conducted our experiments on a workstation equipped with an NVIDIA RTX 5090.
E. More Results
E.1. LLM Judge
We evaluate the robustness of our results to the choice of LLM-based judge by comparing two independent judges. Across
all answerer models and difficulty splits, we observe that the relative rankings and absolute scores are highly consistent,
indicating that the evaluation is not sensitive to the specific judge used.
Table 10 reports oracle-retriever results scored by two judges: GPT-5-mini and GLM-4.7. Differences between the two
judges are small across all models, for both the standard and hard subsets, and do not alter the overall conclusions. This
suggests that our findings are stable with respect to judge selection and are not driven by idiosyncrasies of a particular
evaluation model.
The 0.2% QS score achieved by the No-Evidence baseline corresponds closely to the 0.3% of questions in the dataset labeled
asAbstention, for which the No-Evidence baseline consistently abstains.
E.2. Effect of Retrieval Top-K
We study the effect of retrieval depth by sweeping the number of retrieved memory items ( K) for the all-MiniLM-L6
text-embedding baseline, using Qwen3-VL-8B as the answerer and GPT-5-mini as the judge. Figure 6 reports ATM as a
function ofK.
Overall performance is stable across a wide range of Kvalues. ATM varies only marginally between K= 2 andK= 20 ,
with no monotonic improvement as Kincreases. This indicates that the model already captures most relevant evidence at
relatively small retrieval depths, and that simply increasingKdoes not yield consistent gains.
The slight degradation observed at larger Ksuggests diminishing returns from additional retrieved items, likely due to
the introduction of irrelevant or weakly related memories that increase contextual noise. These results motivate the use of
4https://github.com/OSU-NLP-Group/HippoRAG
5https://github.com/AkariAsai/self-rag
21

Long-Term Personalized Referential Memory QA
Figure 6.ATM (GPT-5-mini) vs. retrieval top-K for the all-MiniLM-L6 text-embedding baseline (Qwen3-VL-8B answerer).
moderate retrieval depths in our main experiments, balancing retrieval coverage and answer stability.
E.3. Indexing cost
Index construction time increases substantially with model size: all-MiniLM-L6-v2 ,Qwen3-Embedding-0.6B ,
Qwen3-Embedding-4B , andQwen3-VL-Embedding-2B require approximately 10 seconds, 1 minute, 20 minutes,
and 40 minutes, respectively. Reranking adds further overhead, with the 0.6B and 4B rerankers taking approximately 10
minutes and 50 minutes, respectively.
F. Error Analysis
We provide some error case analysis here with the oracle retriever setup. We use the error case of the best overall performing
GPT-5.
F.1. Case 1: Failure to Update Memory Over Time (Figure 1(b))
In the oracle setting, both the outdated booking confirmation and the updated invoice email are provided to the answerer.
This setup is designed to explicitly isolate whether the model can correctly prioritize newer memory entries over obsolete
ones, thereby exposing limitations in temporal memory updating.
Question: How much did I pay for my hotel during my recent trip to Portugal?
Ground-truth Answer: C842.97
Model Response (GPT-5):“The cost for the two hotels you stayed in Porto are C408 and C445.26; together is C853.26.”
This error arises because the model relies on the earlier booking confirmation instead of incorporating the updated invoice
information. In other words, the failure mode reflects an inability to perform temporal memory update: the model does not
correctly reconcile conflicting memories by preferring the most recent and authoritative record.
F.2. Case 2: Location Aliasing Under Geocoding Noise
In addition to the common failure in Figure 1(c), where the assistant must infer missing location information by linking
emails with images using temporal reference, we highlight a different challenge unique to real-world personalized memory
corpora:noisy GPS/geocoding metadata.
Figure 7 shows two photos taken within minutes during the same dining event. The correct restaurant is Caf ´e Boh `eme;
however, one image is incorrectly geocoded/parsing-mapped to “Pasha Kebab” despite sharing the same time window and
visual context. In this setting, the metadata is not merely ambiguous across candidates—it iswrongdue to GPS inaccuracy.
The failure mode arises when the model over-trusts the corrupted location field instead of reconciling evidence across nearby
22

Long-Term Personalized Referential Memory QA
(a)Location: Pasha Kebab, Sligo, Time: 2023-01-30,
20:12:10
(b)Location: Caf ´e Boh `eme, Sligo, Time: 2023-01-30,
20:18:15
Figure 7.Failure case caused by location ambiguity across visually similar dining scenes.
timestamps and visual similarity to detect the inconsistency and recover the true location.
We refer to this aslocation aliasing under geocoding noise: a single event is split into conflicting place names due to
erroneous coordinate-to-POI mapping.
Question:I remember having a very nice meal in Sligo, where I had beef and fish. What was the name of that restaurant?
Ground-truth Answer:Caf ´e Boh `eme
Model Response (GPT-5):Pasha Kebab
This failure case reflects a realistic scenario in which GPS metadata is noisy or insufficiently discriminative. The model
over-relies on coarse location cues and selects the wrong restaurant name. Correct resolution requires integrating higher-level
visual and semantic signals—such as food presentation, and dining ambience—to infer which the correct name of the place.
The error highlights the model’s limited ability to perform evidence-based disambiguation when metadata is ambiguous and
multiple plausible memories coexist.
G. Prompts
Here, we provide the prompt for the answer generation and the LLM-judge.
Answerer Prompt
You are a memory QA assistant. Use ONLY the provided evidence to answer. If the evidence is insufficient, answer
“Unknown”. Respond with only the answer. If the question asks to recall or list items (photos/emails/videos),
respond with the corresponding evidence IDs only, comma-separated, with no extra text.
Question:{{question}}
Evidence:
{{evidence}}
Provide the answer based solely on the evidence.
23

Long-Term Personalized Referential Memory QA
LLM Judge Prompt
You are an evaluator, and you are given a task to evaluate a model predictions with a given question. Let’s follow the
instructions step by step to make a judgement.
1. As the first step, you need to check whether the prediction was really answering the question.
2. If the model prediction does provide a meaningful answer, judge whether the model Prediction matches the
ground truth answer by reasoning according to the following steps:
2.1: Always assume the ground truth is correct.
2.2: Pay attention to theses special cases:
a. If the ground truth answer contains numbers, the value of “accuracy” is true only if numbers in ground truth and
numbers in model predictions match very well; in case of math questions, “accuracy” is true only if the numbers in
model predictions EXACTLY matches the numbers in ground truth;
b. If the ground truth answer contains time, and/or time range, “accuracy” is “true” only if if times and time ranges
in ground truth and model predictions match very well.
c. If the ground truth answer contains a set of objects, “accuracy” is “true” if the model prediction covers most of
the objects in the ground truth; however, “accuracy” if “false” if the model prediction has a lot of objects that are not
in the ground truth.
d. If the ground truth is something similar to “I don’t know”, “accuracy” is “true” only if the model prediction also
implies the similar thing.
2.3: Even if the prediction statement is reasonable, if it conflicts with or does not match the ground truth, “accuracy”
should be “false”.
2.4: “Accuracy” is true if the ground truth information is covered by the prediction. The prediction is allowed to
provide more information but should not be against the ground truth. If it is hard to decide whether the prediction
matches ground truth, “accuracy” should be “false”.
Think step by step following the instructions above, and then make a judgment. Respond with only a single JSON
blob with an “explanation” field that has your short (less than 100 word) reasoning steps and an “accuracy” field
which is “true” or “false”.
Question:{{question}}
Ground truth:{{answer}}
Prediction:{{prediction}}
24