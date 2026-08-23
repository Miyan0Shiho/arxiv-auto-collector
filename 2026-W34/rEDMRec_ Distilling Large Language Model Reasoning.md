# rEDMRec: Distilling Large Language Model Reasoning into an Editable Experience Memory for Recommendation

**Authors**: Minh Hoang Nguyen, Tung Le, Huy Tien Nguyen

**Published**: 2026-08-19 14:17:34

**PDF URL**: [https://arxiv.org/pdf/2608.18952v1](https://arxiv.org/pdf/2608.18952v1)

## Abstract
Large language models can improve recommendation quality by reasoning explicitly over user history and candidate items - for example, extracting a user's preferences or explaining why one item fits better than another - rather than mapping history directly to a ranked list. This reasoning, however, is expensive to repeat on every ranking request and, once produced, is typically consumed once and discarded, leaving it neither reusable across future requests nor easy to inspect or correct as user tastes drift. Our insight is that reasoning does not need to be regenerated at every call if it can instead be compressed once into a compact, structured memory that a lightweight model retrieves from. We propose rEDMRec, which distills a teacher LLM's reasoning into four typed, editable experience channels - long-term preference, short-term context, item-perception, and counterfactual hard-negative comparisons - maintained by an LLM memory controller that performs Add/Delete/Modify/Keep operations and refines entries via K-agent debate. A lightweight student LLM then ranks candidates purely by retrieving from this memory, without invoking the teacher again, decoupling online inference cost from reasoning depth. Across ML-1M, Amazon Beauty, and Steam and ten student backbones, rEDMRec improves HR@1 over zero-shot, few-shot, and RAG on every backbone, and over GraphRAG on most backbones, with Impv up to 13.3% vs. the second-best baseline on ML-1M. Channel ablations show that short-term context is the only channel that helps consistently across capacity tiers, whereas long-term, item-perception, and counterfactual contributions are capacity-dependent (and can reverse on the strongest students); debate-based memory optimization lowers bank duplication by 7.4 percentage points while raising downstream HR@1 by up to +0.029 over six optimization epochs.

## Full Text


<!-- PDF content starts -->

rEDMRec: Distilling Large Language Model Reasoning into an
Editable Experience Memory for Recommendation
Minh Hoang Nguyena,b, Tung Lea,band Huy Tien Nguyena,b,∗
aFaculty of Information Technology, University of Science, Ho Chi Minh City, Vietnam
bVietnam National University, Ho Chi Minh City, Vietnam
ARTICLE INFO
Keywords:
LLM-based recommendation
reasoning distillation
experience memory
memory-augmented agents
multi-agent debate
retrieval-augmented generationABSTRACT
Large language models (LLMs) can improve recommendation quality by reasoning explicitly over
user history and candidate items – for example, extracting a user’s preferences or explaining why
one item fits better than another – rather than mapping history directly to a ranked list. This
reasoning, however, is expensive to repeat on every ranking request and, once produced, is typically
consumedonceanddiscarded,leavingitneitherreusableacrossfuturerequestsnoreasytoinspector
correct as user tastes drift. Our insight is that reasoning does not need to be regenerated at every
call if it can instead be compressed once into a compact, structured memory that a lightweight
model retrieves from. We propose rEDMRec, which distills a teacher LLM’s reasoning into four
typed,editableexperiencechannels–long-termpreference,short-termcontext,item-perception,and
counterfactualhard-negativecomparisons–maintainedbyanLLMmemorycontrollerthatperforms
Add/Delete/Modify/Keep operations and refines entries via𝐾-agent debate. A lightweight student
LLM (3B–20B parameters) then ranks candidates purely by retrieving from this memory, without
invoking the teacher again, decoupling online inference cost from reasoning depth. Across ML-
1M, Amazon Beauty, and Steam and ten student backbones, rEDMRec improves HR@1 over zero-
shot, few-shot, and RAG on every backbone, and over GraphRAG on most backbones (exceptions:
Llama 3.1 8B and GPT OSS 20B), with Impv up to13.3%vs. the second-best baseline on ML-1M,
followingtherelative-improvementprotocolofRDRec(Wangetal.,2024b).Channelablationsshow
thatshort-termcontextistheonlychannelthathelpsconsistentlyacrosscapacitytiers,whereaslong-
term, item-perception, and counterfactual contributions are capacity-dependent (and can reverse on
thestrongeststudents);debate-basedmemoryoptimizationlowersbankduplicationby7.4percentage
points while raising downstream HR@1 by up to+0.029over six optimization epochs.
1. Introduction
Large language models (LLMs) are increasingly
used as recommenders – via prompting, instruction
tuning, collaborative-signal fusion, or generative item
prediction(Wuetal.,2024;Linetal.,2025;Baoetal.,2023;
Liao et al., 2024; Hou et al., 2024) – and, more recently, as
explicitreasonersoveruserhistoryandcandidateitems(Wei
etal.,2022).AnLLMcanextractpreferences,judgeitemfit,
orcontrastacandidateagainstahardnegative,thenusethat
reasoning to guide ranking. This paper studies a concrete
bottleneck that follows from that capability: once such
reasoninghasbeenproducedforauser,howcanitbereused
acrossfuturerankingrequests,ratherthanregeneratedfrom
scratch every time?
Reasoning-augmented recommenders face a tension
between reasoning depth and inference cost. Zero-shot and
few-shot prompting, and retrieval-augmented generation
(RAG) over raw interaction history (Lewis et al., 2020),
are cheap but skip explicit preference-level reasoning, so
ranking remains opaque and brittle under short histories.
Closer to our setting, ReasoningRec (Bismay et al.,
2025) uses a teacher LLM to synthesize user profiles,
∗Corresponding author
24C15049@student.hcmus.edu.vn(M.H. Nguyen);
lttung@fit.hcmus.edu.vn(T. Le);ntienhuy@fit.hcmus.edu.vn(H.T.
Nguyen)
ORCID(s):0009-0004-1384-3856(M.H. Nguyen);0000-0002-9900-7047
(T. Le);0000-0002-9948-1048(H.T. Nguyen)item descriptions, and human-interpretable explanatory
reasoning, then instruction-tunes a smaller model on
those traces; R2Rec (Zhao et al., 2025) similarly builds
interaction-of-thought chains and internalizes them with
supervised fine-tuning and reinforcement learning; and
𝑅4ec (Gu et al., 2025) iterates actor reasoning with a
reflection model that critiques and refines preference/item
knowledge before feeding a recommendation backbone.
These lines treat reasoning primarily as a per-request
or training-time signal: competence lives in regenerated
traces or model weights, cannot be updated entry-by-entry
when new interactions arrive or low-quality reasoning
accumulates, and revision requires another costly reasoning
loop or retraining rather than a targeted memory edit.
Relateddistillationmethodscompressrationalesorprompts
intosmallergenerators(Wangetal.,2024b;Lietal.,2023),
yet still treat the distilled artifact as a fixed model rather
than as a typed bank. The remaining challenge is therefore
architectural: keep the benefit of teacher-level reasoning
while making that reasoning reusable across requests
and editable over time, without retraining the backbone
whenever the underlying knowledge must change.
As illustrated in Figure 1, consider a user whose history
isdominatedbyfamilyandanimationtitles(e.g.,ToyStory,
The Lion King,Shrek,Finding Nemo) and who must rank
FrozenagainstTheDarkKnight.Iftherecommenderfocuses
solely on explicit item titles or undifferentiated history text,
it struggles to surface the subtle connection across these
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 1 of 25
arXiv:2608.18952v1  [cs.IR]  19 Aug 2026

rEDMRec: Reasoning Distillation into Experience Memory
Figure 1:Example of four-channel experience memory distilled
from a user’s history for rankingFrozenvs.The Dark Knight.
Offline, teacher reasoning is stored as typedlt/st/ip/cf
entries; online, a frozen student LLM retrieves those channels
and produces the recommendation list without calling the
teacher.
interactions – that the user consistently prefers light fam-
ily entertainment over dark crime – and has no structured
place to store a hard-negative contrast that would pushThe
Dark Knightdown when that pattern holds. In this case,
the latent preference is not a single keyword but a typed
bundle of signals: long-term taste (stable family/animation
preference; dislike of dark crime tone), short-term session
context(franchise-orbuddy-friendlytitles),item-perception
thatgroundswhyFrozenmatchesthehistorywhileTheDark
Knightmismatches, and a counterfactual that records when
the dark-action alternative would have ranked higher. Ex-
isting reasoning-augmented methods either regenerate such
comparisonsperrequestorabsorbthemintoweights,sothey
cannot keep these four signals as independently retrievable
and independently editable memory entries across future
ranking calls.
We address this challenge with rEDMRec. The basic
idea is illustrated in Figure 1: instead of asking an LLM
to both reason and rank on every inference call, we use
the LLM as ateacher reasoning generatorand distill its
outputs into a structured, retrievable, and updatable Experi-
enceMemorythatalightweightstudentconsumesatranking
time.Concretely,theteacherproducesfourtypedexperience
signals – long-term preference, short-term context, item
perception, and counterfactual hard-negative comparisons
– and a distillation adapter normalizes each signal into a
channel-indexed memory entry (vector store for the first
three channels; hybrid vector–graph store for counterfac-
tuals). At inference, afrozenstudent (3B–20B) retrieves
the top-𝑚entries per channel, composes a ranking prompt,
and scores candidates without re-invoking the teacher, so
accuracy gains are attributable to the memory rather than
to student-parameter adaptation. A remaining difficulty is
thatabankfilledoncefromteacherextractionsaccumulates
near-duplicates and low-specificity entries as interactions
grow.InspiredbyTraining-FreeGRPO(Youtu-AgentTeam,
2025), which steers a frozen policy by maintaining an ex-
periential knowledge library with Add/Delete/Modify/Keep
operations rather than gradient updates, our second con-
tribution is an LLM memory controller that applies the
same edit operators – optionally guided by𝐾-agent debateand an arbiter – to revise the recommendation experience
bankoveroptimizationepochs.Together,thesecomponents
separate an expensive, infrequent reasoning-compression
process from a cheap, frequent retrieval-and-rank process.
WeevaluaterEDMReconML-1M,AmazonBeauty,and
SteamwithtenstudentLLMs(3B–20B),comparingagainst
zero-shot, few-shot, RAG, and GraphRAG (Edge et al.,
2024).rEDMRecimprovesHR@1overzero-shot,few-shot,
and RAG on every backbone, and over GraphRAG on most
backbones (exceptions: Llama 3.1 8B and GPT OSS 20B),
reaching Impv= 13.3%vs. GraphRAG on Qwen2.5 3B
under the RDRec relative-improvement protocol (Wang
et al., 2024b) (Section 5.1). Channel ablations show
that short-term context is the only consistently helpful
channel across capacity tiers, whereas long-term, item-
perception, and counterfactual effects are capacity-
dependent(Section5.2).Ateacher-distillationstudyfurther
shows that bank duplicate rate is a leading indicator of
downstream gain, bounded by student capacity. Finally,
debate-based optimization lowers bank duplication by7.4
percentage points while raising downstream HR@1 by up
to+0.029over six epochs, validating the controller as a
functional component of the architecture (Section 5.4).
This paper makes the following contributions:
1.An editable, channel-structured experience
memorythatdistillsteacherLLMreasoningintofour
typed, independently retrievable and independently
editablechannels,ratherthanasingleundifferentiated
reasoning trace (Section 3.6).
2.A debate-based LLM memory controllerthat
revises the bank after each student prediction
via Add/Delete/Modify/Keep operations guided
by ranking reward models and𝐾-agent debate,
with bank-quality gains shown to propagate into
ranking accuracywithout updating the frozenstudent
(Sections 3.6, 3.8; Section 5.4).
3.Acomprehensiveempiricalstudyacrosstenstudent
backbonesandthreedatasets,includingchannelabla-
tionsthatexplainwhenablatingachannelcanimprove
accuracy,ateacher-distillationstudyisolatingteacher
quality from student capacity, and a qualitative case
study of entry-level edits over optimization epochs
(Sections 5.1–5.5).
2. Related Work
2.1. LLM-based Recommendation
Before LLM recommenders, neural models already en-
coded dual-scale user interest, review text, and cold-start
evaluation – but they stored that structure in parameters or
inthecorpus,notinaneditableexperiencebank.Sequential
news recommenders such as Co-NAML-LSTUR (Nguyen
et al., 2025) jointly learn multi-view item encodings with
long-andshort-termuserrepresentations,showingthatsep-
arating durable taste from recent browsing improves rank-
ing even without an LLM. Review-based models such as
RRS (Nguyen et al., 2024) replace ID-only collaborative
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 2 of 25

rEDMRec: Reasoning Distillation into Experience Memory
filtering with deep encoders over user-written text, so item
semantics enter ranking through review content rather than
through a retrieved, typed memory entry. Complementary
dataset work such as ViHoRec (Nguyen, 2026; Nguyen and
Thiet,2025)showsthat,onsparseVietnamesehotelinterac-
tionswithatemporalcold-startsplit,neighborhoodmethods
can outperform learned latent-factor models on users with
short histories. These lines motivate the same signal types
that rEDMRec stores as channels – long-term preference,
short-term context, and item-level text – yet they cannot
persist teacher-generated reasoning or Add/Delete/Modify
a typed entry after a ranking failure. rEDMRec keeps that
factorization, but as a non-parametric bank that a frozen
student retrieves from, rather than as a neural user/item
tower.
Recent surveys organize LLM recommenders into
prompting, tuning, collaborative fusion, generative
recommendation, and system-enhancement paradigms (Wu
et al., 2024; Lin et al., 2025; Wang et al., 2024a; Liu et al.,
2025; Li et al., 2024). Prompting and instruction-following
methodscastrankingasnatural-languagegenerationorzero-
shotordering(Gaoetal.,2023;Zhangetal.,2023;Houetal.,
2024; Yue et al., 2023; Lyu et al., 2024), while alignment
frameworks such as TALLRec and LLaRA adapt LLMs to
recommendationwithefficientfine-tuning(Baoetal.,2023;
Liao et al., 2024). A parallel line injects collaborative or
ID structure into the language space – CoLLM, BinLLM,
TokenRec, and related models encode interaction signals
as text-like or tokenized representations (Zhang et al.,
2025a; 2024b; Qu et al., 2024) – and enhancement
methods use LLMs for graph augmentation, tool use, or
query generation around a classical backbone (Wei et al.,
2024; Zhao et al., 2024; Han et al., 2025). Across these
paradigms, any intermediate reasoning (when present)
remains either ephemeral prompt context or knowledge
absorbed into parameters. rEDMRec instead materializes
teacher reasoning as a non-parametric, typed memory
that survives across sessions and can be revised without
re-tuning the student.
2.2. Reasoning, Distillation, and Preference
Utilization
A narrower line makes LLM reasoning itself the object
of design. ReasoningRec (Bismay et al., 2025) synthesizes
user profiles, item descriptions, and explanatory rationales
with a teacher LLM, then instruction-tunes a smaller model
for both prediction and human-interpretable explanation –
an extraction-then-fine-tune pipeline rather than a durable
memory architecture. R2Rec (Zhao et al., 2025) samples
interaction chains, builds interaction-of-thought traces,
and internalizes them with SFT and RL; LatentR3(Zhang
et al., 2025b) similarly reinforces latent reasoning inside
the model, while SPRec (Gao et al., 2025) uses self-play
to debias generative recommenders.𝑅4ec (Gu et al., 2025)
pushestowardSystem-2deliberationbypairinganactorthat
proposespreference/itemknowledgewithareflectionmodel
that judges and triggers refinement until the knowledgeis deemed rational, then injects the refined text into a
recommendation backbone – still a per-case reasoning
loop rather than a persistent, editable experience bank.
Distillation work such as RDRec, POD, and LEADER
compresses rationales or teacher signals into smaller
recommendationmodels(Wangetal.,2024b;Lietal.,2023;
Liu et al., 2024). Retrieval baselines (RAG, GraphRAG)
ground ranking in raw history or graph summaries without
teacher-compressed experience (Lewis et al., 2020; Edge
et al., 2024). Relative to these methods, rEDMRec neither
stops at one-shot feature utilization nor freezes reasoning
into weights: it distills reasoning into a channel-typed bank
with entry-level Add/Delete/Modify/Keep.
2.3. Memory-Augmented Agents and
Recommendation Memory
External memory lets LLM agents operate beyond a
singlecontextwindow.MemGPT(Packeretal.,2023)pages
context like an OS virtual-memory manager; Generative
Agents (Park et al., 2023) maintain a retrieve-and-reflect
memory stream for long-horizon behavior. Closest to our
controller design, Training-Free GRPO (Youtu-Agent
Team, 2025) improves a frozen LLM by iteratively
distilling experiential knowledge into a non-parametric
library updated with Add/Delete/Modify/Keep operations
– a training-free alternative to gradient-based GRPO. In
recommendation, AutoMR retrieves stored experiences
for generative ranking (Wang et al., 2025), long-term
planners model durable taste (Shi et al., 2024), and
related work motivates separating long- versus short-
term interest (Zheng et al., 2024; Zhang et al., 2024a),
item-level semantics (Ren et al., 2024; Zhang et al., 2026),
hard-negative contrast (Song et al., 2026; Li et al., 2026),
and user-controllable profiles (Woźniak et al., 2025).
These designs retrieve history, profiles, or generic agent
traces; they are not organized as teacher-distilled, four-
channel recommendation experience with debate-driven
bank maintenance. rEDMRec adopts the Training-Free
GRPO principle of editing an external experience library
instead of model weights, but specializes the schema to
recommendation signal types and pairs each channel with a
matching store (vector vs. hybrid graph–vector).
2.4. Multi-Agent Debate and Self-Refinement
Iterative critique improves LLM outputs without addi-
tional supervised labels. Self-Refine (Madaan et al., 2023)
has a model revise its own text from self-feedback; multi-
agent debate (Du et al., 2023) improves factuality by hav-
ing several instances propose and critique answers; and in
recommendation,𝑅4ec (Gu et al., 2025) couples actor and
reflectionmodelstorefinepreference/itemknowledgebefore
backbone prediction. These lines evaluate a single artifact
– one answer, one document, or one knowledge string per
case – and do not measure how repeated refinement of
a growing collection affects duplicate accumulation or re-
trievalqualityatbankscale.rEDMRecappliescritique-and-
revise to the experience bank:𝐾debating personas critique
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 3 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 1
Notation used in Section 3.
Symbol Meaning
𝑢,𝐻𝑢 user; chronological interaction
history
,𝑀(𝑖)item catalog; metadata text of item𝑖
𝐶𝑢⊂,𝑐20-item candidate set; a candidate
in𝐶𝑢
𝑖+,𝑖−,̂𝑖positive target; hard-negative
contrast; predicted item
𝑥𝑢,𝐷𝐾 ranking prompt;𝐾in-context
demonstrations (few-shot)
𝜃,𝑃𝑆 generic LLM parameters; student
likelihood underLLM𝑆
Enc(⋅),simshared text encoder; cosine similarity
𝐪𝑢,𝑑query embeddingEnc(prompt(𝑢,𝐶𝑢));
embedding dim.
,𝑘channel set{lt,st,ip,cf}; a channel
,𝑝,𝑟𝑝 extraction-pass set
{pref,ctx,reas,cf}; a pass; raw
output
𝐸={𝐸𝑘}𝑘∈ experience memory bank (Eq. 8)
𝐸𝑢
𝑘⊂𝐸𝑘 channel-𝑘entries for user𝑢after
metadata filter
𝑒=(𝜏,𝐯𝑒,𝜇)memory entry: text, embedding
𝐯𝑒=Enc(𝜏), metadata
Adapt,route𝑘 distill routed fields into𝑒𝑘; select
fields for channel𝑘
snapshot(𝐸)truncated text rendering of the bank
forLLM𝐶
𝐵={(𝜏,𝑘,𝑢)}insight batch (teacher or arbiter)
consumed byLLM𝐶
𝑜,Applyedit ops
{Add,Delete,Modify,Keep};
commit𝑜to𝐸
𝑚,𝑅𝑘(𝑢,𝐶𝑢)retrieval depth; top-𝑚entries from
𝐸𝑢
𝑘(Eq. 11)
̂𝐿𝑢,rankstudent ranked list; 1-based position
of𝑖+in̂𝐿𝑢
𝑟(𝑢)post-prediction reward vector
(Eq. 13)
𝐾,𝑛𝑟,𝑇#debate agents; rounds/epoch;
optimization epochs
,𝑔𝑗,LLM𝑗 debate transcript; critique of agent
𝑗;𝑗-th debate agent
̃ 𝑒,𝑛max,𝑈arbiter-proposed entry; max
commits/case; case batch
LLM𝑇,LLM𝑆,LLM𝐶,
LLM𝐴teacher, frozen student, controller,
arbiter
a case, an arbiter synthesizes revisions, and the controller
commits Add/Delete/Modify/Keep operations, with bank-
level duplicate rate and specificity linked to downstream
HR@1 (Section 5.4).
3. Method
We formalize next-item ranking as conditional gener-
ation (Section 3.1), restate the dominant prompting and
retrieval paradigms in the same notation to make the tech-
nical gap precise (Section 3.2), then specify rEDMRec’s
components as LLM operators over a typed non-parametric
memory (Sections 3.4–3.8). Table 1 collects the notation
used throughout.3.1. Problem Formulation
We cast next-item recommendation asconditional gen-
eration: given user𝑢, history𝐻𝑢, and a candidate set𝐶𝑢, a
language model with parameters𝜃assigns a score to each
candidate𝑐∈𝐶𝑢by the likelihood of emitting𝑐under a
constructed prompt. The predicted item is
̂𝑖=argmax
𝑐∈𝐶𝑢𝑃𝜃(𝑐∣context(𝑢)),(1)
and is evaluated against the held-out positive𝑖+using
HR@𝑘,NDCG@𝑘,andMRR(Section4.4).Methodsdiffer
in what enterscontext(𝑢); we make this explicit below.
3.2. Preliminaries: Prior Paradigms as
Conditional Generation
Zero-/few-shot prompting(Brown et al., 2020) con-
ditions the LLM on a natural-language prompt𝑥𝑢and𝐾
in-context demonstrations𝐷𝐾= {(𝑥𝑗,𝑦𝑗)}𝐾
𝑗=1, with no
parameter update:
𝑃FS(𝑐∣𝑢)=𝑃𝜃(𝑐∣𝑥𝑢,𝐷𝐾), 𝑥𝑢=prompt(𝐻𝑢,𝑀(𝐶𝑢)).
(2)
𝐾=0recovers zero-shot. Every token of𝑥𝑢and𝐷𝐾is re-
encoded by𝜃at every request.
RAG(Lewis et al., 2020) retrieves a top-𝑘evidence set
𝑍and conditions generation on it:
𝑃RAG(𝑐∣𝑢)=𝑃𝜃(𝑐∣𝑥𝑢,𝑍), 𝑍=Top-𝑘(𝑥𝑢).(3)
In our RAG baseline,𝑍ranges overrawinteraction/review
records and is recomputed for every(𝑢,𝐶𝑢).
GraphRAG(Edge et al., 2024) builds a corpus graph,
partitions it into communities{𝑆𝑙}with LLM-written sum-
maries𝜎(𝑆𝑙),andanswersbyamap–reduceovercommunity
answers𝑎𝑙:
𝑃GRAG(𝑐∣𝑢)=LLM(𝑥𝑢,𝑐,{𝑎𝑙}𝐿
𝑙=1), 𝑎𝑙=LLM(𝑥𝑢,𝜎(𝑆𝑙)).
(4)
{𝜎(𝑆𝑙)}is static once built and is not typed, per-user, or
editable at the granularity of a single fact.
Equations (2)–(4) share a property that motivates our
design: the conditioning set (𝐷𝐾,𝑍, or{𝜎(𝑆𝑙)}) is either
rebuilt/rescored at every request, or, once built, exposes no
operator for targeted, entry-level edits. Section 3.3 replaces
this with a persistent structure𝐸built off the inference
critical path and equipped with an explicit edit operator
(Section 3.6).
3.3. Overview
rEDMRecfactorizesrankingintoanofflineconstruction
ofatypedexperiencememory𝐸andanonline,teacher-free
generative lookup, as illustrated in Figure 2:
𝑃𝑆(𝑐∣𝑢)=𝑃𝑆(
𝑐|||𝑥𝑢,⋃
𝑘∈𝑅𝑘(𝑢,𝐶𝑢))
,
𝐸←Apply(𝐸, 𝑜),(5)
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 4 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Figure 2:Overall architecture of rEDMRec on a concrete case (targetFrozen, hard negativeThe Dark Knight), aligned
with Sections 3.4–3.8.①LLM Teacher as Knowledge Extractor: frozenLLM𝑇emits preference / perception / evidence /
counterfactual traces; Knowledge Distillation (Adapt) writes four-channel entries that the Memory ControllerLLM𝐶commits
via Add/Delete/Modify/Keep.②Editable Experience Memory Bank𝐸= {𝐸lt,𝐸st,𝐸ip,𝐸cf}. Candidate Filter forms𝐶𝑢with
sharedEnc.③LLM Student as Ranker: frozenLLM𝑆retrieves top-𝑚entries per channel and ranks without calling the teacher.
④Experience Memory Optimization: Reward Models𝑟(𝑢)and Debate and Arbiter (LLM𝐴) revise𝐸whileLLM𝑆stays fixed.
where𝑃𝑆denotes generation under the frozen student
LLM𝑆,𝑅𝑘(Eq. 11) is a deterministic top-𝑚dense lookup
(notaper-requestmarginalizedsetasinEq.(3)),andApply
commits edit ops𝑜produced offline by teacher extraction
(Section 3.5) and distillation (Section 3.6), or online by
debate-based optimization after each student prediction
(Section 3.8). This is the central contrast with Eqs. (2)–(4):
reasoning cost is paid when writing𝐸, the frozen student
only retrieves, and memory edits – not weight updates –
amortize future ranking.
Concretely, Figure 2 decomposes the pipeline into
the Method subsections that follow.①LLM Teacher as
Knowledge Extractor (Section 3.5): given(𝑢,𝐻𝑢,𝑖,𝑀),
LLM𝑇runs four extraction passes and Knowledge
DistillationAdapt(Section 3.6.1) normalizes each routed
signal into a channel-indexed entry𝑒= (𝜏,𝐯𝑒,𝜇),
which the Memory ControllerLLM𝐶commits with
Add/Delete/Modify/Keep.②Editable Experience Memory
Bank (Section 3.6):𝐸= {𝐸lt,𝐸st,𝐸ip,𝐸cf}stores stable
taste, session context, candidate-grounded perception,
and counterfactual hard-negative edges as independently
retrievabletypedsnippets.Beforeonlineranking,Candidate
Filter (Section 3.4) forms𝐶𝑢and sharesEncwith memory
indexing.③LLM Student as Ranker (Section 3.7): frozen
LLM𝑆encodes the current query, retrieves the top-𝑚
entries per channel, and produces ̂𝐿𝑢without invoking
LLM𝑇or updating student weights.④Experience Memory
Optimization (Section 3.8): Reward Models𝑟(𝑢)score ̂𝐿𝑢
against𝑖+,thenDebateandArbiterLLM𝐴proposerevisionsthatAdaptandApplywritebackinto𝐸,sofutureretrievals
improve whileLLM𝑆stays fixed.
3.4. Candidate Filter
Motivation:a generative student cannot score the full
catalogat each request, and retrieval plus memory index-
ing must share one dense text space rather than operating
over raw strings.Design:candidate sets𝐶𝑢are formed by
alightweightrecency/popularity/retrieval-scorefilterover
(not a learned projector). The same encoder maps any text
span to a𝑑-dimensional retrieval vector,
𝐯=Enc(text),sim(𝐯,𝐯′)=𝐯⊤𝐯′
‖𝐯‖‖𝐯′‖,(6)
withEnca sentence-transformer (Reimers and Gurevych,
2019).𝐯isreusedformemoryindexing(Eq.9)andretrieval
(Eq. 11).Advantage:𝐶𝑢is formed without a learned pro-
jector, and one sharedEnclets𝐸be queried directly with
candidate or user vectors, with no separate alignment step.
3.5. LLM Teacher as Knowledge Extractor
Motivation:conflating “understand the user” (stable)
with “score this candidate list” (per-request) forces the
former to be redone every request; rEDMRec queries the
teacher only for the former, and only toextract, never to
rank.Design:followingReasoningRec(Bismayetal.,2025)
and R2Rec (Zhao et al., 2025), which show that structured
preference profiles, item-level perceptions, and explanatory
rationales improve recommendation when distilled from
a strong LLM, teacherLLM𝑇runs four extraction passes
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 5 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 2
Teacher extraction passes𝑝∈: output fields and routed
channel(s)𝑘(Section 3.6.1).
𝑝Key output fields Routed to𝑘
preflong_term_preferences,dislikes,
short_term_preferenceslt,st
ctxuser_history_perception,candidate_perceptionip
reassteps[1..5],reasoning_summaryip
cfcounterfactual_condition/outcome,rationalecf
𝑝∈, each a prompted generation,
𝑟𝑝=LLM𝑇(prompt𝑝(𝑢,𝑖,𝐻𝑢,𝑀)),
𝑝∈.(7)
whose output fields are later routed to one or more memory
channels𝑘∈byAdapt(Table 2, Section 3.6.1); the
routing is not one-to-one, since two passes jointly pop-
ulate the item-perception channel. Prompt templates and
example one-line outputs for all four channels are given in
Appendix I; full Beauty/Steam prediction traces (history,
candidates,𝑅𝑘, ranked list) appear in Appendix J. Unlike
ReasoningRec/R2Rec,whichconsumethesetracesastrain-
ing targets or one-shot prompt features, we never update
studentparametersfrom𝑟𝑝:eachfieldiscommittedintothe
editable bank𝐸of Section 3.6.
Preference extraction (pref).Simulates a recurrent,
batch-by-batch update over𝐻𝑢(oldest→newest): for each
historybatchitre-estimatesalong-termpreferencestateand
thecurrent-batchshort-terminterest,plusanexplicitdislike
list; the final long-term/dislike fields route to channelltand
the short-term field routes to channelst.
Context extraction (ctx).For every history item and
candidate, produces three layers – an objective factual
description, a first-person “as this user” comment, and
(candidates only) key phrases; both the history-perception
andcandidate-perceptionfieldsroutetotheitem-perception
channelip.
Reasoning extraction (reas).Runs a five-step chain-
of-thought that identifies shared themes across liked items,
scores each candidate against them, contrasts the top two
candidates, and states a final recommendation; its summary
field also routes toip, complementingctxwith an explicit
comparative rationale rather than a per-item description.
Counterfactual extraction (cf).Given an anchor (cho-
sen)itemandacontrast(hard-negative)item,produceswhy-
preferred/why-rejectedrationalesandahypotheticalcondi-
tionunderwhichthecontrastitemwouldoutranktheanchor;
routesentirelytochannelcfasagraphedge(Section3.6.2).
Anoptionalsingle-passmulti-agentrefinementcritiques
the𝑟cfoutput with three fixed personas before distillation,
mergingby“lastfullcritiquewins”withnoarbiter;thisisa
lighter-weightprecursortothearbiter-basedoptimizationof
Section3.8,whichinsteadrevisesalready-committedentries
across all four channels using downstream reward signals.
Advantage:separating extraction into four typed passes,
rather than one undifferentiated call, lets Section 3.6.1 editasingleroutedclaimwithouttouchingtheotherchannelsof
the same interaction.
3.6. Editable Experience Memory Bank
Motivation:Eq. (7) yields free-form JSON that cannot
be reused across sessions or revised entry-by-entry once
committed.Design:adapting the training-free experiential-
knowledge library of Training-Free GRPO (Youtu-Agent
Team, 2025) to recommendation, rEDMRec materializes
teacheroutputintoapersistent,typednon-parametricmem-
ory
𝐸={𝐸𝑘}𝑘∈, 𝐸𝑘={𝑒∣𝑒.memory_type=𝑘},(8)
where each entry𝑒= (𝜏,𝐯𝑒,𝜇)carries distilled text𝜏,
embedding𝐯𝑒= Enc(𝜏), and channel-specific metadata𝜇.
Population follows a two-stage commit path –Adapt(field
routingandschemanormalization)andApplyviacontroller
LLM𝐶(batch edit ops) – detailed in Section 3.6.1; channel
semantics and retrieval are in Section 3.6.2.
3.6.1. Knowledge Distillation: Adapter and Memory
Controller
Distillation adapterAdapt.Adaptmaps routed teacher
fields (Table 2) into a typed entry and dispatches it to the
correct physical store:
𝑒𝑘=Adapt(route𝑘({𝑟𝑝}𝑝∈))
=(𝜏𝑘,Enc(𝜏𝑘), 𝜇𝑘), 𝑘∈.(9)
Hereroute𝑘(⋅)selectsandconcatenatesthefield(s)assigned
to channel𝑘(many-to-one:pref→{lt,st};ctx,reas→ip;
cf→cf).Adaptthen(i)normalizes𝜏𝑘intoachannelschema
(𝜇𝑘holds typed fields such aslong_term_preferences,
dislikes,steps,anchor_item), (ii) encodes𝜏𝑘with the
sharedEncof Eq. (6), and (iii) writes to𝐸lt,𝐸st, or𝐸ip
(Vector database) or, for𝑘=cf, to𝐸cfas a Graph database
edgepair(𝑢)ANCHOR← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← →(𝑖+)∧(𝑢)CONTRAST← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← →(𝑖−)plusarationale
embedding(Listing3).Initialextraction(distill_to_memory)
and debate optimization (Section 3.8) both call the same
Adapt.
Memory controllerLLM𝐶.RawAdaptoutput is not
appendedblindly:controllerLLM𝐶inspectsatextsnapshot
of the bank against a batch of new insights𝐵={(𝜏, 𝑘, 𝑢)}
and emits structured edit ops𝑜, whichApplycommits in
order:
𝑜=LLM𝐶(snapshot(𝐸), 𝐵),
𝑜𝑖∈{ADD,DELETE,MODIFY,KEEP},
𝐸←Apply(𝐸, 𝑜).(10)
Applyimplements each op on the underlying stores:
ADDcallsAdaptthen appends; DELETEremoves by
entry_id; MODIFYre-encodes revised𝜏and replaces
metadata in-place; KEEPis a no-op. These four operators
mirror the experience-library update rule in Training-
Free GRPO (Youtu-Agent Team, 2025), specialized here
to typed recommendation channels rather than general
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 6 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 3
Four experience-memory channels: role, teacher source, stor-
age, and retrieval filter.
𝑘Semantic role Source𝑝Storage / filter
ltStable cross-session
tasteprefvector DB;user_id=𝑢
stSession-level trend /
driftprefvector DB +
timestamp;user_id=𝑢
ipItem impression vs.
userctx,reasvector DB;user_id=𝑢,
item=𝑐
cfContrastive “if...”
edgecfgraph DB + vector
DB;user_id=𝑢
agent rollouts. The controller prompt enforces a maximum
library size, deduplication (merge over add), and≤60-word
actionable entries.Advantage:Eqs. (9)–(10) decouplewhat
to remember (teacher or arbiter) fromhowto commit it;
refinement cost scales with|𝑜|, not|𝐸|.
3.6.2. Four Experience-Memory Channels
We factor experience into four typed channels because
prior LLM and sequential recommenders show that long-
horizon taste (Shi et al., 2024; Wang et al., 2025), recent
sequentialcontext(Zhengetal.,2024;Zhangetal.,2024a),
item-level semantics (Ren et al., 2024; Zhang et al., 2026),
and hard-negative contrast (Song et al., 2026; Li et al.,
2026) each contribute distinct ranking signal – and because
storing them separately lets𝑅𝑘retrieve only the evidence
class needed for a decision. Table 3 summarizes the four
channels. Each𝐸𝑘is independently indexed, independently
retrievable, and independently editable – the ablation in
Section5.2dropsindividual𝑘atinferencewithoutretraining
LLM𝑆.
Long-term preference (lt).Motivated by evidence that
long-horizon preference modeling improves recommenda-
tion beyond next-click prediction (Shi et al., 2024) and that
external memory retrieval is needed when the LLM context
window alone drops long-term history (Wang et al., 2025),
this channel stores durable taste statements distilled from
long_term_preferencesanddislikes(prefpass), optionally
with supportingreasoning. Entries are user-global: retrieval
over𝐸𝑢
lt= {𝑒∈𝐸lt∣𝜇.user_id=𝑢}answers “what
does this user generally like/dislike?” without binding to a
specific candidate.
Short-term context (st).Motivated by sequential
recommenders that show recent interactions dominate
next-item prediction (Zheng et al., 2024) and that jointly
modeling long- and short-term interests outperforms either
alone (Zhang et al., 2024a), this channel captures transient
interest fromshort_term_preferences(prefpass) plus
optionalcontext_reasoning. Each entry carries a timestamp
in𝜇, enabling the bank to represent tastedriftwithin a
session or across recent batches whileltremains stable.
Retrieval filters onuser_idonly.
Item perception (ip).Motivated by work showing that
item-text representation quality drives LLM recommen-
dation (Ren et al., 2024) and that token-centric attention{
"user_id": "...",
"anchor_preference":
"prefers animation,
family-friendly tone",
"target_item": "Frozen",
"contrast_item":
"The Dark Knight",
"counterfactual_condition":
"if user prefers dark,
action-heavy stories",
"counterfactual_outcome":
"The Dark Knight would
rank higher",
"rationale_text": "...",
"embedding": [],
"timestamp": "..."
}
Figure 3:Distilled𝑒cf: the graph-edge instantiation of Eq.(9).
under-models item-level collaborative relations (Zhang
et al., 2026), this is the most granular channel: (i) per-
candidate impressions fromcandidate_perception(ctx), (ii)
per-history-item descriptions fromuser_history_perception
(ctx), and (iii) comparative reasoning from the five-step
chain andreasoning_summary(reas). Candidate-specific
entries additionally indextarget_movie_idin𝜇, so retrieval
for(𝑢,𝑐)returns impressions scoped to𝑐rather than
unrelated items.
Counterfactual (cf).Motivated by recent LLM
recommenders that separate hard negatives from noisy
negatives (Song et al., 2026) and that use self-hard
negatives to sharpen preference learning (Li et al., 2026),
this channel stores hard-negative contrast rationales as
typed graph edges: for anchor item𝑖+and contrast𝑖−,
𝜇recordswhy_anchor_preferred,why_contrast_rejected,
counterfactual_condition, andcounterfactual_outcome.
Dense retrieval runs overrationale_textembeddings; a
graph-augmented pass appends any user-specific edges not
surfaced by vector search. Listing 3 shows the instantiated
schema.
Retrieval.At inference, the student queries each
active channel with a shared query embedding𝐪𝑢=
Enc(prompt(𝑢,𝐶𝑢))and returns the top-𝑚entries by dense
similarity:
𝑅𝑘(𝑢,𝐶𝑢)=Top-m
𝑒∈𝐸𝑢
𝑘sim(𝐪𝑢,𝐯𝑒),(11)
where𝐸𝑢
𝑘⊆ 𝐸𝑘applies the filters in Table 3 (candidate
scope𝑐is applied only forip), andsimis Eq. (6). Default
𝑚=5per channel.Advantage:typed channels let𝑅𝑘return
onlytheevidenceclassneededforarankingdecision–stable
taste (lt), recent drift (st), candidate fit (ip), or boundary
conditions (cf) – rather than one undifferentiated reasoning
blob.
3.7. LLM Student as Ranker
Motivation:if ranking gains required adapting student
parameters, it would be unclear whether accuracy came
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 7 of 25

rEDMRec: Reasoning Distillation into Experience Memory
from the editable experience memory or from parameter
adaptation;moreover,aper-backboneadaptedstudentwould
need re-adaptation for every backbone, undermining the
claimthat𝐸helpsanyteacher–studentpair.Design:student
LLM𝑆is a frozen pretrained LLM (3B–20B in our study).
At inference it ranks solely by prompting with retrieved
experience – never a fresh teacher call and never a weight
update – following Eq. (5):
̂𝑖=argmax
𝑐∈𝐶𝑢𝑃𝑆(𝑐∣𝑢)
=argmax
𝑐∈𝐶𝑢𝑃𝑆(
𝑐|||𝑥𝑢,⋃
𝑘∈𝑅𝑘(𝑢,𝐶𝑢))
.(12)
Here𝑥𝑢= prompt(𝑢,𝐶𝑢)and⋃
𝑘𝑅𝑘are concatenated into
thestudentcontext;LLM𝑆isheldfixedacrossbankupdates.
Unlike Eq. (3), retrieval is deterministic, and no student
objective is optimized.Advantage:gains over zero-/few-
shot and RAG – and over GraphRAG on most backbones
(Section5.1)–areattributabletothecontentandeditability
of𝐸, not to student adaptation – the same frozenLLM𝑆
improves when𝐸improves (Section 5.4), and the protocol
transfers across teacher–student pairs without per-student
training (Section 5.3).
3.8. Experience Memory Optimization
Motivation:a bank populated once from teacher extrac-
tion can accumulate generic or conflicting entries; after the
studentranksacase,themismatchbetweenthepredictedlist
and the ground-truth target is a direct signal that𝐸should
berevised.Design:mirroringTraining-FreeGRPO(Youtu-
Agent Team, 2025)’s loop of rollout→reward→semantic
advantage→library edit – but specialized to recommenda-
tion ranking and enriched with𝐾-agent debate – we update
𝐸after each student prediction: the student emits a ranked
list under Eq. (12), reward models score that list against𝑖+,
𝐾debating agents critique the case conditioned on those
rewards, an arbiter synthesizes revised experience entries,
and the same distillation controller commits them. This
closesalooppredict→reward→debate→edit→𝐸without
changingLLM𝑆.
3.8.1. Reward Models
Motivation:a single scalar (e.g., only Hit@1) is too
coarse for debate agents to diagnosewhya ranking failed
– missing the top item, burying it just outside the top-𝑘, or
placing it deep in the list require different memory edits.
Design:given the student’s ordered list ̂𝐿𝑢=(𝑐(1),…,𝑐(𝐿))
and ground-truth target𝑖+, letrank ∈ {1,…,𝐿} ∪ {∅}
be the 1-based position of𝑖+in̂𝐿𝑢(∅if absent). We
pack four complementary IR-style rewards in[0,1]into the
debate/arbiter prompt:
𝑟(𝑢)=(𝟏[rank=1],𝟏[rank≤𝑘],
min(1,1∕rank),1∕log2(rank+1))∈[0,1]4,
(13)
with all components zero ifrank = ∅(default𝑘=10).
Designrationale.ThefourcomponentsareHR@1,HR@𝑘,Algorithm 1Experience Memory Optimization after stu-
dent prediction (one epoch)
Require:case batch𝑈, frozen studentLLM𝑆, agents
{LLM1,…,LLM𝐾}, arbiterLLM𝐴, rounds𝑛𝑟
1:for𝑢∈𝑈do
2: Rank ̂𝐿𝑢with frozenLLM𝑆via Eq. (12)
3:𝑟(𝑢)←reward models on( ̂𝐿𝑢,𝑖+)⊳Eq. (13)
4:←∅
5:forround=1,…,𝑛𝑟; agent𝑗=1,…,𝐾do
6:←‖LLM𝑗(persona𝑗,,𝑢,̂𝐿𝑢,𝑟(𝑢))⊳Eq. (14)
7:end for
8:{̃ 𝑒1,…,̃ 𝑒𝑛}←LLM𝐴(,𝑢,̂𝐿𝑢,𝑟(𝑢))⊳Eq. (15)
9:for̃ 𝑒∈{̃ 𝑒1,…,̃ 𝑒𝑛}do
10:𝑒←Adapt(̃ 𝑒);𝐸←Apply(𝐸,𝑜)⊳Eqs. (9),(10)
11:end for
12:end for
reciprocal rank, and DCG-style position discount: they flag
strict top-rank failure, near-misses, and graded credit when
𝑖+appearsbutnotfirst.Importantly,𝑟(𝑢)isnotatrainingloss
onLLM𝑆andisnotusedtoargmaxamongagentproposals;
itconditionsthenatural-languagecritiquesothateditstarget
the observed ranking failure mode.Advantage:the same
four signals transfer across datasets and students because
theydependonlyon( ̂𝐿𝑢,𝑖+),keepingtheoptimizationloop
student-agnostic.
3.8.2. Debate and Arbiter after Each Prediction
Design lineage.The procedure combines the
training-free experience-library update of Training-Free
GRPO (Youtu-Agent Team, 2025) with iterative self-
critique(Madaanetal.,2023)andmulti-personadebate(Du
et al., 2023), but applies them at memory-bank granularity
and triggers them from the student’s post-prediction reward
vector (Section 3.8.1).
Group computation.Over𝑛𝑟rounds,𝐾fixed-persona
agents append free-form critiques to a shared transcript:
𝑔𝑗=LLM𝑗(persona𝑗,, 𝑢,̂𝐿𝑢, 𝑟(𝑢)),
←‖𝑔𝑗.(14)
No entry is committed during Eq. (14). After𝑛𝑟𝐾turns, a
singlearbitercallsynthesizestheexperiencesettocommit:
{̃ 𝑒1,…,̃ 𝑒𝑛}=LLM𝐴(, 𝑢,̂𝐿𝑢, 𝑟(𝑢)),
𝑛≤𝑛max.(15)
Eq.(15)isanLLMsynthesis,notaprogrammaticvoteoran
argmax𝑗over{𝑔𝑗}by reward.
Optimizing.Algorithm 1 runs, for each case: student
ranking(Eq.12)→rewardmodels(Eq.13)→debate/arbiter
(Eqs. 14–15)→commit viaAdaptandApply. Section 5.4
tracks bank-quality signals and downstream HR@1/MRR
over𝑇such epochs.
Reuse of Knowledge Distillation.Each synthesized̃ 𝑒
is committed by thesameoperators as regular extraction
(Section 3.6.1):𝑒=Adapt(̃ 𝑒)(Eq. 9) and𝐸←Apply(𝐸,𝑜)
(Eq. 10). Optimization does not updateLLM𝑆; it only re-
vises𝐸.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 8 of 25

rEDMRec: Reasoning Distillation into Experience Memory
4. Experimental Setup
4.1. Datasets
Weevaluateonthreerecommendationdatasetsthatspan
the explicit-implicit feedback spectrum.ML-1M(Harper
and Konstan, 2015) provides 1–5 star explicit ratings and
is a standard benchmark for LLM-based recommendation.
Amazon Beauty(Ni et al., 2019) provides explicit ratings
and review text but is substantially sparser per user, which
stresses the memory bank’s ability to compensate for weak
collaborativesignal.Steam(KangandMcAuley,2018)isan
implicit-feedbackdataset:itlogswhichgamesauserplayed
rather than an explicit rating, so every logged interaction is
treated as an equally-weighted positive signal (rating fixed
to1.0, positive threshold0.0) instead of being thresholded
from a graded rating scale. For all three datasets we apply a
𝑘-corefilter(𝑘=20onML-1M;𝑘=5onBeautyandSteam),
build a chronological train/validation/test split, and con-
struct20-candidaterankingsamples(1positive,19sampled
negatives). Dataset statistics after filtering are reported in
Appendix H (Table 20).
4.2. Baselines
We compare against four prompting/retrieval baselines
and rEDMRec, formalized in Eqs. (2)–(4) (Section 3.2):
Zero-shotandFew-shot(Brown et al., 2020) (𝐾=0and
𝐾>0in Eq. 2);RAG(Lewis et al., 2020) (Eq. 3), which
retrieves raw historical interactions or reviews rather than
distilled reasoning; andGraphRAG(Edge et al., 2024)
(Eq. 4), which retrieves from a co-occurrence/knowledge
graph built over items.
4.3. Models
Unless noted otherwise, the teacher isgpt-5.4-mini.
We evaluate ten open student backbones spanning 3B–20B
parameters: Qwen2.5 3B, Llama 3.1 8B (Touvron et al.,
2023), Gemma-4-12B (Gemma Team, 2024), Minimax
M2.5, Mixtral 8x7B (Jiang et al., 2024), Qwen3-14B (Yang
et al., 2024), DeepSeek-R1-Distill-Qwen-14B (DeepSeek-
AI, 2025), Phi-4, Llama 4 Scout, and GPT OSS 20B. Each
student is usedfrozen(Section 3.7), so that cross-backbone
gains isolate the contribution of the editable experience
memory. The teacher-distillation study (Section 5.3)
additionally fixes the student togpt-5-mini(strong) or
Qwen2.5 3B (small) while varying the teacher across seven
backbones, to isolate the teacher’s contribution from the
student’s.
4.4. Metrics and Protocol
We report Hit Rate at rank𝑘(HR@𝑘,𝑘∈ {1,5,10}),
Normalized Discounted Cumulative Gain (NDCG@𝑘,
𝑘∈ {5,10}), and Mean Reciprocal Rank (MRR),
all computed over the fixed 20-candidate set per
evaluation sample. For RQ1 tables we report Impv
(%), the relative HR@1 improvement of rEDMRec
over the second-best baseline on the same student
(Impv = (Ours − SecondBest)∕SecondBest × 100),
following RDRec (Wang et al., 2024b), together witha McNemar𝑝-value on HR@1 vs. that second-best
method (approximate2×2contingency reconstructed
from the table HR@1 rates at the full held-out sizes
𝑛ML−1M=49893,𝑛Beauty=1460,𝑛Steam=1460;∗marks
𝑝<0.05when rEDMRec is ahead). Every (model, method,
dataset) cell is evaluated on thefull held-out test split
(chronological train/validation/test; 20 candidates per
sample with sampling seed42), not a pilot subsample. The
cross-dataset comparison in Section 5.1 (Amazon Beauty,
Steam), the bank-scale analysis in Section 5.2, and the𝑘-
EPOCH and number-of-agents curves in Section 5.4 follow
the same full-split evaluation protocol at each reported
dataset, bank scale, and epoch/agent-count setting.
5. Results
We organize results around four research questions:
does distilling reasoning into memory improve ranking
over prompting-only and retrieval-only baselines (RQ1,
Section 5.1)? which memory channel drives that
improvement (RQ2, Section 5.2)? does teacher quality
causally affect bank quality and downstream gain (RQ3,
Section 5.3)? and does debate-based memory optimization
measurably improve bank quality and downstream ranking
(RQ4, Section 5.4)? We close with a qualitative case study
of how individual memory entries evolve over training
(Section 5.5).
5.1. RQ1: Does rEDMRec Improve Ranking
Across Students and Datasets?
Table 4 reports HR@𝑘, NDCG@𝑘, and MRR for
four representative students spanning our capacity range
(Qwen2.5 3B, Llama 3.1 8B, Mixtral 8x7B, Qwen3-14B)
onML-1M;thefullten-modeltableisgiveninAppendixA.
rEDMRec improves HR@1 over Zero-shot, Few-shot, and
RAG for every student. Following RDRec (Wang et al.,
2024b), Table 4 reports Impv (%) vs. the second-best
baseline (typically GraphRAG): the largest relative gain is
on the smallest student (Qwen2.5 3B, Impv= 13.3%over
GraphRAG), consistent with structured memory helping
most when parametric capacity is limited. Llama 3.1 8B
is the clearest failure case – Impv is negative because
GraphRAG remains ahead (Impv= −11.1%) – which
we attribute to weak instruction-following rather than a
deficiency of the memory itself (Section 4.3; Section 6);
GPT OSS 20B similarly trails GraphRAG slightly (Impv
= −3.3%), so the claim relative to GraphRAG ismost, not
all, students.
Table 5 reports Zero-shot vs. rEDMRec on Amazon
Beauty and Steam for all ten student models under HR@1,
NDCG@10, and MRR (full five-method matrices in Ap-
pendix B–C). The same ordering as on ML-1M holds:
rEDMRec improves HR@1 over Zero-shot for every model
onbothdatasets,andNDCG@10/MRRriseinlockstepex-
ceptontheweakestLlama3.18Bbackbone,whereranking
beyond top-1 remains flat. Absolute scores are lower than
on ML-1M – as expected for sparser Amazon Beauty and
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 9 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 4
Main results on ML-1M for four representative students (full ten-model table in Appendix A). Impv (%) is the relative HR@1
gain of rEDMRec over thesecond-bestbaseline on the same student,Impv=(Ours−SecondBest)∕SecondBest×100, following
RDRec (Wang et al., 2024b).𝑝is the exact McNemar𝑝-value for rEDMRec HR@1 vs. the second-best baseline (full held-out
𝑛=49893; approximate contingency from the table HR@1 rates).∗marks𝑝<0.05with rEDMRec ahead. Best inbold, second-best
underlined ; rEDMRec rows labeled in bold.
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Qwen2.5 3B Zero-shot 0.12 0.28 0.38 0.20 0.23 0.18 – –
Few-shot 0.14 0.30 0.40 0.21 0.24 0.20 – –
RAG 0.13 0.29 0.39 0.20 0.23 0.19 – –
GraphRAG 0.15 0.31 0.41 0.22 0.25 0.20 – –
rEDMRec
(ours)0.17∗0.35 0.45 0.25 0.28 0.23 +13.3∗<0.001∗
Llama 3.1 8B Zero-shot 0.07 0.15 0.23 0.12 0.14 0.12 – –
Few-shot 0.08 0.16 0.24 0.12 0.15 0.13– –
RAG 0.08 0.16 0.24 0.12 0.14 0.12 – –
GraphRAG0.09 0.17 0.250.12 0.15 0.13– –
rEDMRec
(ours)0.08 0.17 0.25 0.13 0.16 0.13 -11.1<0.001
Mixtral
8x7BZero-shot 0.24 0.47 0.66 0.35 0.40 0.35 – –
Few-shot 0.26 0.49 0.68 0.37 0.42 0.36 – –
RAG 0.25 0.48 0.67 0.36 0.41 0.35 – –
GraphRAG 0.27 0.50 0.69 0.38 0.42 0.37 – –
rEDMRec
(ours)0.28∗0.52 0.71 0.39 0.44 0.38 +3.7∗<0.001∗
Qwen3-14B Zero-shot 0.26 0.50 0.69 0.38 0.43 0.37 – –
Few-shot 0.28 0.52 0.71 0.40 0.45 0.39 – –
RAG 0.27 0.51 0.70 0.39 0.43 0.38 – –
GraphRAG 0.29 0.53 0.72 0.40 0.45 0.40 – –
rEDMRec
(ours)0.30∗0.55 0.73 0.42 0.47 0.41 +3.4∗<0.001∗
implicit-feedback Steam – yet Impv (%) vs. the second-best
baseline(GraphRAG)isagainlargestonthesmalleststudent
(Qwen2.5 3B: Impv= 23.6%on Beauty and21.5%on
Steam;both𝑝<0.05),consistentwithmemorycompensating
forweakcollaborativesignalwhenper-userhistoryisshort;
mid-capacity Beauty students show smaller Impv (about9–
13%) that is not significant at𝑛=1460.
5.2. RQ2: Which Memory Channel Matters?
To isolate each channel’s contribution, we remove one
memory channel at a time and reportΔHR@1 relative
to the full-memory model. This ablation uses a separate
panel of seven API-served backbones chosen to span four
capacity tiers – strong (gpt-5-mini), mid (gpt-5.4-mini,
Qwen3-32B, Minimax M2.5), saturated (GPT-OSS-120B),
and weak (Llama 3.3 70B, Llama 3.1 8B) – rather than
the ten local open-weight students used for the main
comparison in Section 5.1; note that a backbone name can
appear in both this ablation panel and the teacher panel of
Section 5.3 (e.g.,gpt-5.4-mini), where it plays a different
role (ablated student vs. teacher) in a separate experiment.
Table 6 summarizes the resulting pattern across the four
capacity tiers. Short-term context is the only channel that
is consistently important: removing it hurts every tier, from
a strong student (gpt-5-mini,Δ=−0.04) down to weak
Groq-served Llama students (Δ ≈ −0.01to−0.02). The
long-term preference, item-perception, and counterfactualchannels show areversedablation on the strongest student:
removingthemimprovesHR@1by+0.03to+0.04,whereas
removingthecounterfactualchannelhurtsthemid-capacity
student (Δ=−0.04) and has little effect on the saturated
120B-parameter student, which appears to ignore the bank
altogether (full-memoryΔ≈0.00for that tier).
We consider three, non-exclusive explanations for the
reversed sign on strong students, in decreasing order of
the evidence we can bring to bear with the current instru-
mentation.First,channelredundancy:long-termpreference
statements often restate information already present in the
raw history block of the prompt, so a strong student that
already attends well to raw history gains nothing extra and
insteadpaysasmall“distraction”costfortheredundanttext.
Second,generic,low-specificityentries:item-perceptionen-
tries produced early in the bank’s lifecycle tend to be long
and only loosely actionable (Section 5.5 quantifies this di-
rectly via a specificity score). Third,conflicting signals:
a counterfactual edge can push a plausible but ultimately
incorrect candidate above the true target when its hypo-
thetical condition partially matches the current user. Only
themid-capacitystudent,whichcannotyetextractthesame
information unaided from raw history, benefits from the
counterfactualchannelunconditionally;thefactthatthesign
of this effect depends on student capacity, rather than being
fixed, is itself evidence against treating any single channel
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 10 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 5
Cross-dataset generalization on Amazon Beauty and Steam (full held-out test split, 20 candidates/sample, seed 42). ZS =
Zero-shot; Ours = rEDMRec (bold). Impv (%) is the relative HR@1 gain of rEDMRec over thesecond-bestbaseline on the same
student,Impv=(Ours−SecondBest)∕SecondBest×100, following RDRec (Wang et al., 2024b).𝑝= McNemar𝑝-value for Ours
HR@1 vs. the second-best baseline (𝑛Beauty=1460,𝑛Steam=1460;∗if𝑝<0.05). Complete five-method matrices are in Appendix B–C.
Dataset Model HR@1(ZS) HR@1(Ours) NDCG@10(ZS) NDCG@10(Ours) MRR(ZS) MRR(Ours) Impv (%)𝑝
Amazon
BeautyQwen2.5 3B 0.0920.136∗0.2180.2620.1800.224 +23.6∗0.037∗
Llama 3.1 8B 0.0620.0710.1800.1800.1620.162 -4.10.830
Gemma-4-12B 0.1400.1780.2900.3280.2520.290 +12.70.166
Minimax M2.5 0.1520.1870.3080.3430.2640.290 +10.00.247
Mixtral
8x7B0.1640.2020.3200.3580.2820.311 +11.00.188
Qwen3-14B 0.1760.2110.3380.3730.2940.329 +8.80.269
DeepSeek-R1
Distill-Qwen-14B0.1880.2230.3500.3850.3120.329 +8.30.280
Phi-4 0.1700.2050.3320.3930.2880.340 +9.00.264
Llama 4 Scout 0.1820.2170.3440.4050.3000.344 +8.50.275
GPT OSS 20B 0.1820.1990.3440.3700.3000.317 -0.51.000
SteamQwen2.5 3B 0.1060.158∗0.2240.2760.1800.232 +21.5∗0.035∗
Llama 3.1 8B 0.0660.0760.1800.1800.1620.162 -7.30.584
Gemma-4-12B 0.1700.2080.3200.3580.2760.314 +7.20.356
Minimax M2.5 0.1860.2240.3440.3820.2920.321 +6.70.394
Mixtral
8x7B0.2020.2440.3600.4020.3160.349 +8.00.276
Qwen3-14B 0.2180.2560.3840.4220.3320.370 +5.80.392
DeepSeek-R1
Distill-Qwen-14B0.2340.2720.4000.4380.3560.375 +5.40.426
Phi-4 0.2100.2480.3760.4430.3240.382 +6.00.411
Llama 4 Scout 0.2260.2640.3920.4590.3400.388 +5.60.421
GPT OSS 20B 0.2260.2450.3920.4210.3400.359 -2.00.797
Table 6
Channel importance by capacity tier (ΔHR@1 vs. full memory).
Negative = beneficial channel; positive = reversed ablation
(Section 5.2).
Channel Strong Mid Saturated Weak
Short-term context−0.04 −0.04 ≈−0.01 −0.01–−0.02
Long-term
preference+0.03 0.00 ≈0.00 +0.01
Item-perception+0.04 0.00 ≈0.00slight
Counterfactual+0.03 −0.04weak≈0.00
Full memory−0.01 −0.03 ≈0.00weak
asuniversallyusefulorharmful.Figure4showsthispattern
holds across all seven backbones in the ablation panel, and
Figure 5 shows that the reversed channels flip sign as the
bankgrows from the sparse-bank anchor (𝐵=189) toward
105–106entries,becauseadenserbankdilutestheredundant
and generic entries that drive the reversal at small𝐵(bank
size is independent of the full-test evaluation protocol in
Section 4.4). Per-backboneΔplots, the MRR heatmap, and
thefullnumericablationmatrixaredeferredtoAppendixG.
5.3. RQ3: Does Teacher Quality Causally Affect
Downstream Gain?
Holdingthestudentfixedandvaryingtheteacherisolates
the teacher’s contribution from the student’s. Table 7 fixes
the student togpt-5-mini(strong) across seven teachers;
Table 8 repeats the comparison with Qwen2.5 3B (small).
w/o Short-Term w/o Long-Term w/o Item-Perc.w/o Counterfactualw/o Memory
Channel removedGPT-5-mini
GPT-5.4-mini
GPT-OSS-120B
Qwen3-32B
Minimax M2.5
Llama-3.3-70B
Llama-3.1-8BStudent model-0.06 -0.06 -0.08 -0.05 -0.10
-0.05 -0.05 -0.07 -0.05 -0.08
-0.01 -0.01 -0.01 -0.01 -0.02
-0.05 -0.05 -0.07 -0.05 -0.08
-0.05 -0.05 -0.07 -0.05 -0.08
-0.02 -0.02 -0.02 -0.02 -0.04
-0.02 -0.02 -0.02 -0.02 -0.04Memory channel ablation: ΔHR@1 across student models (n=30,000)
-0.10-0.050.000.050.10
ΔHR@1Figure 4:Channel ablation heatmap:ΔHR@1 for each of the
seven ablation-panel backbones (rows, Section 5.2) and four
memory channels (columns). Green cells indicate a reversed
ablation (removal helps); red cells indicate the channel is
beneficial.
Across both tables, a lower bank duplicate rate is a leading
indicator of downstream gain:gpt-5.4-minihas the lowest
duplicate rate among the strongest teachers (12.4%) and
the largestΔHR@1 for the strong student (+0.060), while
Llama 3.1 8B Instant – the only teacher below 14B param-
eters in this comparison – has by far the highest duplicate
rate(22.8%)andthesmallestgain(+0.015),consistentwith
aweakteacherproducinggeneric,repetitiveentries(e.g.,re-
peated“user likes drama”statements without item-specific
detail) that carry little retrieval value. The relationship is
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 11 of 25

rEDMRec: Reasoning Distillation into Experience Memory
189 200 500 1K 5K 10K 30K
Teacher extractions B-0.06-0.04-0.020.000.020.040.060.080.10Contribution (−ΔHR@1 vs. Full)
+0.04 +0.04+0.04+0.05+0.06 +0.06 +0.06
-0.03 -0.03-0.02-0.01+0.04+0.06+0.06
-0.04 -0.04-0.03-0.01+0.05+0.07+0.07
-0.03 -0.03-0.02-0.02+0.03+0.05+0.05(a) Contribution vs. bank scale B
100 200 500 1K 2K 5K 10K
Neval (log10)0.040.050.060.070.08Contribution (−ΔHR@1)
(b) Stability at B=29K
N_eval=500 fixed in (a). B≤200: LT/IP/CF reversed; B≥5K all −ΔHR@1. (b) bank B=29,502 fixed; error bars σ/√N.Short-Term Long-Term Item-Perception Counterfactual
Figure 5:Per-channel contribution (−ΔHR@1) as bank scale
𝐵grows from the sparse-bank anchor (𝐵=189) to106entries
(panel a), and stability of the estimate as evaluation sample
sizegrowsunderthefull-testprotocolatfixedbankscale(panel
b). Note:𝐵is bank size, not the evaluation split size.
Table 7
Teacher-distillation effectiveness, fixed student:gpt-5-mini
(strong). Zero-shot baseline HR@1=0.28, MRR=0.403.
Teacher Dup.%↓ΔHR@1↑ΔMRR↑HR@1
gpt-5.4-mini12.4+0.060 +0.0687 0.340
Qwen3 32B (131k) 11.5 +0.055 +0.0620 0.335
gpt-5-mini 14.1 +0.050 +0.0580 0.330
Llama 3.3 70B
(128k)10.5 +0.045 +0.0520 0.325
GPT OSS 120B
(128k)9.8+0.040 +0.0450 0.320
Minimax M2.5 13.2 +0.040 +0.0480 0.320
Llama 3.1 8B Instant 22.8 +0.015 +0.0180 0.295
not perfectly monotonic, however: GPT OSS 120B has the
lowest duplicate rate of any teacher (9.8%) yet only a mid-
dlingΔHR@1 (+0.040), which we attribute to a student-
capacity ceiling –gpt-5-minicannot fully exploit the addi-
tional verbosity of a 120B-parameter teacher’s bank. This
ceiling is sharper for the small student: Table 8 shows the
GPT OSS120B bankdropsbelowthe moreconcise Qwen3
32B bank once the student itself is small (Qwen2.5 3B),
even though GPT OSS 120B produces the least duplicated
bankofthetwo.Together,thesetwotablessupportacausal
chain in which teacher quality first improves bank quality
(lower duplication), and bank quality only converts into
downstream gain up to a ceiling set by the student’s own
capacity to use additional bank detail.
5.4. RQ4: Does Debate-Based Memory
Optimization Improve Bank Quality and
Downstream Ranking?
Wenextaskwhetherthedebate-and-arbiteroptimization
procedure (Section 3.8) is doing useful work, rather than
merelyaddingcost.Figure6tracksbank-qualitysignalsand
downstream ranking jointly over six𝑘-EPOCHs of debate
(three debate agents, one round per epoch). Bank quality
improves and saturates: the duplicate rate drops by7.4Table 8
Teacher-distillation effectiveness, fixed student: Qwen2.5 3B
(small). Zero-shot baseline HR@1=0.12, MRR=0.18.
Teacher Dup.%↓ΔHR@1↑ΔMRR↑HR@1
gpt-5.4-mini12.4+0.050 +0.0550 0.170
Qwen3 32B (131k) 11.5 +0.045 +0.0500 0.165
gpt-5-mini 14.1 +0.042 +0.0470 0.162
Llama 3.3 70B
(128k)10.5 +0.038 +0.0430 0.158
GPT OSS 120B
(128k)9.8+0.034 +0.0380 0.154
Minimax M2.5 13.2 +0.033 +0.0380 0.153
Llama 3.1 8B Instant 22.8 +0.010 +0.0120 0.130
0 1 2 3 4 5 6
k-EPOCH (debate passes)0.150.200.250.300.35Downstream score
(a) Debate → downstream ranking
Mixtral 8x7B · HR@1
Mixtral 8x7B · MRR
Minimax M2.5 · HR@1
Minimax M2.5 · MRR
Gemma-4-12B · HR@1Gemma-4-12B · MRR
Qwen2.5 3B · HR@1
Qwen2.5 3B · MRR
Mixtral 8x7B · HR@1 (no-debate)0 1 2 3 4 5 6
k-EPOCH (debate passes)1112131415161718Duplicate rate (%)
(b) Debate → bank quality
Duplicate rate % (↓)
Mean reward (↑)
Specificity (↑)
0.00.20.40.60.81.0
Reward / specificity
Figure 6:Debate-based memory optimization vs.𝑘-EPOCH.
(a) Downstream HR@1/MRR for student models plus a no-
debate control. (b) Bank-quality signals: duplicate rate (down
is better) and mean experience reward / specificity (up is
better).
percentagepoints(18.0%→10.6%)whilemeanexperience
reward rises by0.255(0.52→0.78). Downstream ranking
tracksthiscurveratherthanmovingindependentlyofit:the
strongest student in this comparison (Mixtral 8x7B) gains
+0.029HR@1 (0.250→0.279) over the same six epochs,
whileano-debateparaphrasecontrol–whichperturbsentry
wording without the critique-and-revise debate loop – stays
flat,isolatingthedebatemechanism(ratherthananywording
change) as the source of the gain. The smallest student in
this comparison (Qwen2.5 3B) gains only+0.013HR@1
(0.156→0.169) from the identical optimized bank, repro-
ducing the capacity ceiling from Section 5.3 in a different
experiment: most of the gain lands within the first two to
threeepochsforeverystudent,sodebatingpast𝑘=3epochs
is rarely worth the added LLM cost.
A separate sweep over thenumberof debating agents
(𝑘= 1,…,10, one epoch) shows quality rising with the
diversity of critique but saturating, while LLM cost grows
linearlyin𝑘(Table9,Figure7).Thekneeofquality-per-cost
isat𝑘∗=4:increasing𝑘from1to4liftsHR@1by+0.022,
but increasing𝑘from 4 to 10 adds only+0.006for six
additionalLLMcallspercase,whichisnotafavorabletrade
for most deployment budgets. Tabular controller ablations
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 12 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 9
Number-of-debating-agents sweep (𝑘= 1..10), reference stu-
dent Mixtral 8x7B.𝑘∗marks the quality-per-cost knee.
𝑘HR@1↑Specificity↑Dup.%↓Calls/case
1 0.255 0.520 16.0 2
2 0.266 0.608 13.4 3
3 0.273 0.663 11.9 4
4 (𝑘∗) 0.277 0.699 11.05
5 0.279 0.721 10.4 6
6 0.281 0.735 10.0 7
7 0.282 0.744 9.8 8
8 0.282 0.750 9.7 9
9 0.282 0.754 9.6 10
10 0.283 0.756 9.6 11
12345678910
k (debate agents)0.160.180.200.220.240.260.28HR@1
k*=4(a) Downstream vs k
Mixtral 8x7B
Minimax M2.5Gemma-4-12B
Qwen2.5 3B
12345678910
k (debate agents)0.00.20.40.60.81.0Specificity
(b) Bank quality vs k
Specificity (↑) Duplicate % (↓)
12345678910
k (debate agents)246810LLM calls / case
(c) Cost vs k (linear)
LLM calls/case Debate tokens/case
10111213141516
Duplicate rate (%)
5001000150020002500
Tokens / case
Figure 7:Number-of-agents sweep: downstream HR@1, bank-
quality signals, and LLM cost as a function of𝑘.
(fulldebatevs.no-debateparaphrasevs.post-extraction)and
epoch snapshots are collected in Appendix F.
5.5. Qualitative Case Study: How Do Memory
Entries Evolve?
The ablation and debate-optimization results above are
aggregate signals; to make the mechanism concrete, we
directlycomparetheearliestandthelatestpersistedentryper
user in each memory channel, using a deterministic, LLM-
free specificity score (specificity combines concreteness,
genre-termcoverage,lexicaldiversity,andahedge-language
penalty into a single[0,1]score, with no additional LLM
calls).Table10summarizestheresultingbefore/afterdeltas
across users with at least two persisted versions. Three of
the four channels become more specific and less hedged
overtraining;theitem-perceptionchannelchangesthemost
(mean specificity0.405→0.488), moving from generic
fallback language to item-grounded, actionable statements.
The short-term context channel is the exception: its mean
specificitydecreasesslightly (0.433→0.401), which Fig-
ure8andthecasesbelowshowisnotaqualityregressionbut
acompressioneffect–verbosenarrativeentriesarereplaced
by short, conditional “session rules” that are less lexically
diverse by construction but more directly actionable by the
student.
Table 11 shows three representative before→after pairs
drawn from the persisted bank (one long-term, one item-
perception, one short-term).Case A(long-term, user 2)
replaces a single-film hedge (“Very limited data...”) with aTable 10
Bank evolution: specificity before (earliest persisted entry) vs.
after (latest persisted entry) per channel, over users with≥2
versions.
Channel Spec. before Spec. afterΔ
Long-term preference 0.500 0.520+0.021
Short-term context 0.433 0.401−0.032
Item-perception 0.4050.488 +0.084
Counterfactual /
hard-neg.0.478 0.519+0.042
Long-Term
PreferenceShort-Term
ContextItem-Perception Contrastive
Hard-Negative0.00.10.20.30.40.50.6Mean specificity+0.02
-0.03+0.08+0.04Experience-bank specificity: before vs after training
Before After
Figure 8:Mean specificity before vs. after training, by memory
channel.
ranking-readytastestatementthatnamesera,franchisetype,
andanupweightrule,liftingspecificityfrom0.214to0.479.
Case B(item-perception, user 4) converts a vague Hit@1
complaintintoanitem-groundedrulekeyedtoRocky(1976),
with an explicit+30–50%tie-break boost – the largest
single-entryΔSpec.inthesample(+0.500).CaseC(short-
term, user 2) illustrates the compression pattern behind the
negative meanΔon that channel: an empty “no emerging
interests” note becomes a short session rule with diversity
constraints; lexical diversity drops relative to long narrative
entries,butactionabilityrises.Togetherthethreecasesshow
that debate-driven Add/Modify operations do not merely
paraphraseentries–theyaccumulateconfirmedsignalsinto
shorter, more specific, ranking-oriented memory.
6. Discussion and Limitations
Operating-point generalization.The bank-scale anal-
ysis (Section 5.2) and the𝑘-EPOCH and number-of-agents
curves (Section 5.4) span a wide range of settings – bank
sizesupto106entries,uptosixdebateepochs,anduptoten
debatingagents.Apractitioneradoptingaspecificoperating
point(forexample,aspecific𝑘-EPOCHbudgetorbanksize)
in a production system should re-confirm behavior at that
exact point, since a trend measured across a range does not
guarantee identical behavior at every intermediate setting.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 13 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 11
Qualitative before→after memory entries (persisted bank). Spec. is the deterministic specificity score in[0,1].
Case / channel Spec. Before (earliest) After (latest)
A/lt 0.214→0.479“Very limited data: the only rated film is a classic
action-adventure, suggesting a preference for high-
energy, heroic, escapist storytelling...Dislikes: No
explicit dislikes can be inferred...”“Long-term: favors mainstream 1990s action
buddy-cop films – franchise sequels, star-driven
chemistry, high-energy action with comedic in-
terplay. Upweight these attributes in ranking but
require repeat confirmations...”
B/ip 0.350→0.850“The strongest match was only ranked 5th, so
Hit@10 was good but Hit@1/MRR suffered. For
this user, place the best Rocky-like inspirational
drama at rank 1 whenever possible.”“Rocky (1976): User 4 chose Rocky over a higher-
ranked 80s action title, signaling a preference for
1970s character-driven underdog sports dramas.
When ranking, boost similar 1970s character dra-
mas above mainstream 80s action (recommend
+30–50%score in tie scenarios).”
C/st 0.300→0.500“No recent viewing items were provided, so no
emerging short-term interests can be detected.”“Session: prioritize late-80s/90s Hollywood action
comedies with buddy dynamics and franchise en-
tries for top slots; include at least one diverse
alternative per slate to avoid popularity bias.”
Teacher coverage.The main results (Section 5.1) fix
the teacher togpt-5.4-mini; the teacher-distillation study
(Section 5.3) varies the teacher but only against two fixed
students. We have not measured the full teacher×student
cross-product,soitremainsopenwhethertheteacher-quality
effect observed forgpt-5-miniand Qwen2.5 3B holds uni-
formly across all ten students in Table 12.
Explanation faithfulness.rEDMRec’s student can
emit a short explanation grounded in retrieved memory
(Section 3.7), but this paper evaluates ranking quality, not
whether the emitted explanation is faithful to the memory
it cites; a human evaluation of explanation faithfulness and
plausibility, as outlined in our evaluation plan, is left to
future work.
Backbone-dependent returns.The architecture
assumes the student can follow a moderately complex
prompt that concatenates user context, candidate
descriptions, and retrieved memory snippets. Section 5.1
shows this assumption breaks down for at least one
backbone (Llama 3.1 8B), whose negative Impv vs.
GraphRAG we attribute to weak instruction-following
rather than to the memory being unhelpful in principle;
the near-saturated 20B-parameter student likewise trails
GraphRAG slightly. Memory still lifts every student over
Zero-shot/Few-shot/RAG, but not always over GraphRAG
– Appendix K tabulates these failure cells and borderline
controller edits. Practically, the value of rEDMRec’s added
system complexity is highest for small-to-mid capacity
students.
Domainscope.Ourthreedatasetscovermovie,beauty-
product, and game recommendation with English-language
metadata;wehavenottesteddomainswithsubstantiallydif-
ferent item-description structure (e.g., short-video or news
recommendation), and the four-channel schema, in partic-
ular the counterfactual channel, was designed with catalog
items that have stable, comparable attributes in mind.7. Conclusion
Thispaperaddressestheproblemofreusing,ratherthan
repeating,LLMreasoningacrossrecommendationrequests,
by proposing rEDMRec, an architecture that distills teacher
reasoning into a four-channel, editable experience memory
and serves ranking requests from a lightweight student that
onlyretrievesfromthismemory.Thekeyideaistoseparate
an infrequent, expensive reasoning-compression process –
teacher extraction, distillation, and debate-based memory
optimization – from a frequent, cheap inference process,
which decouples recommendation quality from per-request
reasoningcost.AcrosstenstudentLLMsandthreedatasets,
this design improves HR@1, NDCG, and MRR over
zero-shot, few-shot, and RAG on every backbone, and over
GraphRAG on most backbones (exceptions: Llama 3.1 8B
andGPTOSS20B),withthelargestRDRec-styleImpv(%)
vs. the second-best baseline on the smallest students;
our channel-ablation, teacher-distillation, and debate-
optimization studies further show that short-term context
is the only consistently beneficial channel across capacity
tiers (with long-term, item-perception, and counterfactual
effects capacity-dependent), that bank duplication is a
leading indicator of downstream gain up to a student-
capacity ceiling, and that𝐾-agent debate measurably
improves bank quality in a way that propagates into
ranking accuracy rather than being a cosmetic refinement.
Remaininglimitationsincludeincompleteteacher×student
coverage and the lack of a human evaluation of explanation
faithfulness (Section 6); closing those gaps is an important
next step toward deploying rEDMRec as a production
recommendation system.
Data and Code Availability
The preprocessing, training, and evaluation code, to-
gether with the JSON experiment matrices used to produce
every table and figure in this paper, are organized under
therEDMRec/project root (seereadme.mdfor the end-to-end
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 14 of 25

rEDMRec: Reasoning Distillation into Experience Memory
quick-start pipeline). ML-1M, Amazon Beauty, and Steam
are third-party datasets redistributed under their original
licenses; this work releases only derived, de-identified in-
teraction records and memory-bank artifacts.
CRediT authorship contribution statement
Minh Hoang Nguyen:Methodology, Conceptualiza-
tion, Writing – original draft, Writing – review & editing.
Tung Le:Supervision, Supporting, Writing – review &
editing.Huy Tien Nguyen:Supervision, Supporting, Con-
ceptualization, Project administration.
References
Bao,K.,Zhang,J.,Zhang,Y.,Wang,W.,Feng,F.,He,X.,2023. TALLRec:
An effective and efficient tuning framework to align large language
modelwithrecommendation,in:Proceedingsofthe17thACMConfer-
ence on Recommender Systems, pp. 1007–1014. doi:10.1145/3604915.
3608857.
Bismay, M., Dong, X., Caverlee, J., 2025. ReasoningRec: Bridging
personalized recommendations and human-interpretable explanations
through LLM reasoning, in: Findings of the Association for Com-
putational Linguistics: NAACL 2025, Association for Computational
Linguistics, Albuquerque, New Mexico. pp. 8147–8163. URL:https:
//aclanthology.org/2025.findings-naacl.454/.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal,
P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al., 2020.
Languagemodelsarefew-shotlearners.AdvancesinNeuralInformation
Processing Systems (NeurIPS) .
DeepSeek-AI, 2025. DeepSeek-R1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint arXiv:2501.12948 .
Du,Y.,Li,S.,Torralba,A.,Tenenbaum,J.B.,Mordatch,I.,2023.Improving
factualityandreasoninginlanguagemodelsthroughmultiagentdebate.
arXiv preprint arXiv:2305.14325 .
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt,
S., Larson, J., 2024. From local to global: A graph RAG approach to
query-focused summarization. arXiv preprint arXiv:2404.16130 .
Gao, C., Chen, R., Yuan, S., Huang, K., Yu, Y., He, X., 2025. SPRec:
Self-play to debias LLM-based recommendation. arXiv preprint
arXiv:2412.09243 .
Gao,Y.,Sheng,T.,Xiang,Y.,Xiong,Y.,Wang,H.,Zhang,J.,2023. Chat-
REC: Towards interactive and explainable LLMs-augmented recom-
mender system. arXiv preprint arXiv:2303.14524 .
GemmaTeam,2024. Gemma:Openmodelsbasedongeminiresearchand
technology. arXiv preprint arXiv:2403.08295 .
Gu, H., Zhong, R., Xia, Y., Yang, W., Lu, C., Jiang, P., Gai, K., 2025.
𝑅4ec:Areasoning,reflection,andrefinementframeworkforrecommen-
dation systems, in: Proceedings of the Nineteenth ACM Conference on
Recommender Systems, ACM, Prague, Czech Republic. pp. 411–421.
doi:10.1145/3705328.3748068.
Han, D., Song, H., Yi, M.Y., 2025. Rethinking LLM-based recommenda-
tions: A personalized query-driven parallel integration. arXiv preprint
arXiv:2504.11889 QueREC.
Harper, F.M., Konstan, J.A., 2015. The MovieLens datasets: History and
context. ACM Transactions on Interactive Intelligent Systems 5, 1–19.
Hou,Y.,Zhang,J.,Lin,Z.,Lu,H.,Xie,R.,McAuley,J.,Zhao,W.X.,2024.
Largelanguagemodelsarezero-shotrankersforrecommendersystems,
in: European Conference on Information Retrieval, Springer. pp. 364–
381.
Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford,
C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., et al., 2024.
Mixtral of experts. arXiv preprint arXiv:2401.04088 .
Kang,W.C.,McAuley,J.,2018. Self-attentivesequentialrecommendation,
in: IEEE International Conference on Data Mining (ICDM), pp. 197–
206.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela,
D., 2020. Retrieval-augmented generation for knowledge-intensive
NLP tasks, in: Advances in Neural Information Processing Systems
(NeurIPS), pp. 9459–9474.
Li, B., Zheng, B., Wang, X., Zhang, L., Wang, J., Chen, S., Zhao, W.X.,
Wen,J.R.,2026. ImprovingLLM-basedrecommendationwithself-hard
negatives from intermediate layers. arXiv preprint arXiv:2602.17410 .
Li, L., Zhang, Y., Chen, L., 2023. Prompt distillation for efficient LLM-
basedrecommendation,in:Proceedingsofthe32ndACMInternational
Conference on Information and Knowledge Management, pp. 1348–
1357. doi:10.1145/3583780.3615017.
Li, L., Zhang, Y., Liu, D., Chen, L., 2024. Large language models
for generative recommendation: A survey and visionary discussions,
in: Proceedings of the 2024 Joint International Conference on Com-
putational Linguistics, Language Resources and Evaluation (LREC-
COLING 2024), Torino, Italia. pp. 10146–10159.
Liao,J.,Li,S.,Yang,Z.,Wu,J.,Yuan,Y.,Wang,X.,He,X.,2024. LLaRA:
Large language-recommendation assistant, in: Proceedings of the 47th
InternationalACMSIGIRConferenceonResearchandDevelopmentin
Information Retrieval, pp. 1785–1795. doi:10.1145/3626772.3657760.
Lin,J.,Dai,X.,Xi,Y.,Liu,W.,Chen,B.,Zhang,H.,Liu,Y.,Wu,C.,Li,X.,
Zhu,C.,etal.,2025. Howcanrecommendersystemsbenefitfromlarge
languagemodels:Asurvey. ACMTransactionsonInformationSystems
Also arXiv:2306.05817.
Liu, Q., Wu, X., Zhao, X., Zhu, Y., Zhang, Z., Tian, F., Zheng, Y., 2024.
Large language model distilling medication recommendation model.
arXiv preprint arXiv:2402.02803 LEADER.
Liu, Q., Zhao, X., Wang, Y., Wang, Y., Zhang, Z., Sun, Y., Li, X., Wang,
M., Jia, P., Lin, K., et al., 2025. Large language model enhanced
recommender systems: A survey. arXiv preprint arXiv:2412.13432 .
Lyu,H.,Jiang,S.,Zeng,H.,Xia,Y.,Wang,Q.,Zhang,S.,Chen,R.,Leung,
C., Tang, J., Luo, J., 2024. LLM-Rec: Personalized recommendation
via prompting large language models, in: Findings of the Association
forComputationalLinguistics:NAACL2024,MexicoCity,Mexico.pp.
583–612. doi:10.18653/v1/2024.findings-naacl.39.
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S.,
Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al., 2023. Self-
refine: Iterative refinement with self-feedback, in: Advances in Neural
Information Processing Systems (NeurIPS), pp. 46534–46594.
Nguyen, M.H., 2026. Vihorec: A quality-controlled vietnamese hotel
recommendation dataset and cold-start benchmark. arXiv preprint
arXiv:2607.12946 .
Nguyen, M.H., Nguyen, T.T., Ta, M.N., Le, T., Nguyen, H.T., 2025. Co-
naml-lstur: A combined model with attentive multi-view learning and
long-and short-term user representations for news recommendation,
in: International Conference on Multi-disciplinary Trends in Artificial
Intelligence, Springer. pp. 106–119.
Nguyen, M.H., Nguyen, T.T., Ta, M.N., Nguyen, T.M., Nguyen, K.V.,
2024. Rrs: Review-based recommendation system using deep learning
for vietnamese. SN Computer Science 5, 492.
Nguyen, M.H., Thiet, S.N., 2025. Enhancing ocr for sino-vietnamese
language processing via fine-tuned paddleocrv5. arXiv preprint
arXiv:2510.04003 .
Ni, J., Li, J., McAuley, J., 2019. Justifying recommendations using
distantly-labeled reviews and fine-grained aspects, in: Conference on
EmpiricalMethodsinNaturalLanguageProcessing(EMNLP),pp.188–
197.
Packer, C., Fang, V., Patil, S.G., Lin, K., Wooders, S., Gonzalez, J.E.,
2023. MemGPT: Towards llms as operating systems. arXiv preprint
arXiv:2310.08560 .
Park, J.S., O’Brien, J., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S.,
2023. Generative agents: Interactive simulacra of human behavior, in:
ACM Symposium on User Interface Software and Technology (UIST),
pp. 1–22.
Qu, H., Fan, W., Zhao, Z., Li, Q., 2024. TokenRec: Learning to tok-
enize ID for LLM-based generative recommendation. arXiv preprint
arXiv:2406.10450 .
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 15 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Reimers, N., Gurevych, I., 2019. Sentence-BERT: Sentence embeddings
using siamese BERT-networks, in: Conference on Empirical Methods
in Natural Language Processing (EMNLP), pp. 3982–3992.
Ren, X., Wei, W., Xia, L., Su, L., Cheng, S., Wang, J., Yin, D., Huang,
C., 2024. Representation learning with large language models for
recommendation, in: Proceedings of the ACM Web Conference 2024,
pp. 3464–3475. doi:10.1145/3589334.3645458.
Shi, W., He, X., Zhang, Y., Gao, C., Li, X., Zhang, J., Wang, Q., Feng,
F., 2024. Large language models are learnable planners for long-term
recommendation,in:Proceedingsofthe47thInternationalACMSIGIR
ConferenceonResearchandDevelopmentinInformationRetrieval,pp.
1893–1903.
Song, T., Chao, W.S., Liu, H., 2026. Hard vs. noise: Resolving hard-noisy
sample confusion in recommender systems via large language models,
in: Proceedings of the AAAI Conference on Artificial Intelligence, pp.
15743–15751. doi:10.1609/aaai.v40i18.38605.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei,
Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al., 2023.
Llama 2: Open foundation and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
Wang,C.,Zhang,Y.,Zhu,F.,Zhang,J.,Shi,T.,Feng,F.,2025. Leveraging
memoryretrievaltoenhancellm-basedgenerativerecommendation,in:
Companion Proceedings of the ACM on Web Conference 2025, pp.
1346–1350.
Wang, Q., Li, J., Wang, S., Xing, Q., Niu, R., Kong, H., Li, R., Long,
G., Chang, Y., Zhang, C., 2024a. Towards next-generation LLM-
based recommender systems: A survey and beyond. arXiv preprint
arXiv:2410.19744 .
Wang, X., Cui, J., Suzuki, Y., Fukumoto, F., 2024b. RDRec: Rationale
distillation for LLM-based recommendation, in: Proceedings of the
62ndAnnualMeetingoftheAssociationforComputationalLinguistics
(Volume2:ShortPapers),Bangkok,Thailand.pp.65–74. doi:10.18653/
v1/2024.acl-short.6.
Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Ichter,B.,Xia,F.,Chi,E.,
Le,Q.,Zhou,D.,2022. Chain-of-thoughtpromptingelicitsreasoningin
large language models, in: Advances in Neural Information Processing
Systems (NeurIPS), pp. 24824–24837.
Wei, W., Ren, X., Tang, J., Wang, Q., Su, L., Cheng, S., Wang, J., Yin,
D., Huang, C., 2024. LLMRec: Large language models with graph
augmentation for recommendation, in: Proceedings of the 17th ACM
InternationalConferenceonWebSearchandDataMining,pp.806–815.
doi:10.1145/3616855.3635853.
Woźniak, S., Duszenko, J., Kocoń, J., Kazienko, P., 2025. Improving llm-
based recommender systems with user-controllable profiles, in: Com-
panion Proceedings of the ACM on Web Conference 2025, pp. 2102–
2111.
Wu, L., Zheng, Z., Qiu, Z., Wang, H., Gu, H., Shen, T., Qin, C., Zhu, C.,
Zhu,H.,Liu,Q.,Xiong,H.,Chen,E.,2024. Asurveyonlargelanguage
models for recommendation. arXiv preprint arXiv:2305.19860 .
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C.,
Liu, D., Huang, F., et al., 2024. Qwen2 technical report. arXiv preprint
arXiv:2407.10671 .
Youtu-AgentTeam,2025. Training-freegrouprelativepolicyoptimization.
arXiv preprint arXiv:2510.08191 URL:https://arxiv.org/abs/2510.
08191.
Yue, Z., Rabhi, S., Moreira, G.d.S.P., Wang, D., Oldridge, E., 2023.
LlamaRec:Two-stagerecommendationusinglargelanguagemodelsfor
ranking. arXiv preprint arXiv:2311.02089 .
Zhang, J., Xie, R., Hou, Y., Zhao, W.X., Lin, L., Wen, J.R., 2023. Recom-
mendationasinstructionfollowing:Alargelanguagemodelempowered
recommendation approach. arXiv preprint arXiv:2305.07001 .
Zhang, X., He, B., Chen, J., Cui, Z., Ma, C., 2026. From token to item:
Enhancing large language models for recommendation via item-aware
attention mechanism, in: Proceedings of the ACM Web Conference
2026, pp. 6700–6708.
Zhang, X., Li, B., Jin, B., 2024a. Denoising long- and short-term interests
for sequential recommendation, in: Proceedings of the 2024 SIAM
InternationalConferenceonDataMining(SDM),pp.544–552. doi:10.1137/1.9781611978032.63.
Zhang, Y., Bao, K., Yan, M., Wang, W., Feng, F., He, X., 2024b. Text-
like encoding of collaborative information in large language models for
recommendation, in: Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers),
Bangkok, Thailand. pp. 9181–9191. doi:10.18653/v1/2024.acl-long.
497.
Zhang,Y.,Feng,F.,Zhang,J.,Bao,K.,Wang,Q.,He,X.,2025a. CoLLM:
Integratingcollaborativeembeddingsintolargelanguagemodelsforrec-
ommendation. IEEETransactionsonKnowledgeandDataEngineering
Also arXiv:2310.19488.
Zhang,Y.,Xu,W.,Zhao,X.,Wang,W.,Feng,F.,He,X.,Chua,T.S.,2025b.
Reinforced latent reasoning for LLM-based recommendation. arXiv
preprint arXiv:2505.19092 LatentR3.
Zhao, K., Xu, F., Li, Y., 2025. Reason-to-recommend: Using interaction-
of-thoughtreasoningtoenhanceLLMrecommendation. arXivpreprint
arXiv:2506.05069 R2Rec.
Zhao, Y., Wu, J., Wang, X., Tang, W., Wang, D., de Rijke, M., 2024. Let
me do it for you: Towards LLM empowered recommendation via tool
learning. arXiv preprint arXiv:2405.15114 .
Zheng, Z., Chao, W., Qiu, Z., Zhu, H., Xiong, H., 2024. Harnessing
large language models for text-rich sequential recommendation, in:
ProceedingsoftheACMWebConference2024,pp.3207–3216. doi:10.
1145/3589334.3645358.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 16 of 25

rEDMRec: Reasoning Distillation into Experience Memory
A. Full Main-Results Table
Table 12 reports the complete ML-1M main-results matrix underlying Table 4 in Section 5.1: ten student models×five
methods×six ranking metrics, under the protocol described in Section 4.4. Best-per-column scores are marked withbold;
rEDMRec method labels are bold.
Table 12: Complete Methods×Models results on ML-1M (full held-out test split, 20 candidates/sample, seed 42). Impv
(%) is the relative HR@1 gain of rEDMRec over thesecond-bestbaseline on the same student,Impv = (Ours −
SecondBest)∕SecondBest×100, following RDRec (Wang et al., 2024b). Best inbold, second-best underlined ; rEDMRec
rowslabeledinbold.𝑝istheexactMcNemar𝑝-valueforrEDMRecHR@1vs.thesecond-bestbaselineonthesamestudent
(full held-out𝑛ML−1M=49893,𝑛Beauty=1460,𝑛Steam=1460; approximate contingency from the table HR@1 rates).∗marks
𝑝<0.05with rEDMRec ahead.
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Qwen2.5 3B Zero-shot 0.12 0.28 0.38 0.20 0.23 0.18 – –
Few-shot 0.14 0.30 0.40 0.21 0.24 0.20 – –
RAG 0.13 0.29 0.39 0.20 0.23 0.19 – –
GraphRAG 0.15 0.31 0.41 0.22 0.25 0.20 – –
rEDMRec
(ours)0.17∗0.35 0.45 0.25 0.28 0.23 +13.3∗<0.001∗
Llama 3.1 8B Zero-shot 0.07 0.15 0.23 0.12 0.14 0.12 – –
Few-shot 0.08 0.16 0.24 0.12 0.15 0.13– –
RAG 0.08 0.16 0.24 0.12 0.14 0.12 – –
GraphRAG0.09 0.17 0.250.12 0.15 0.13– –
rEDMRec
(ours)0.08 0.17 0.25 0.13 0.16 0.13 -11.1<0.001
Gemma-4-
12BZero-shot 0.20 0.40 0.58 0.30 0.35 0.30 – –
Few-shot 0.22 0.42 0.60 0.32 0.37 0.31 – –
RAG 0.21 0.41 0.59 0.31 0.36 0.30 – –
GraphRAG 0.23 0.43 0.61 0.33 0.38 0.32 – –
rEDMRec
(ours)0.24∗0.46 0.63 0.34 0.39 0.34 +4.3∗<0.001∗
Minimax
M2.5Zero-shot 0.22 0.44 0.62 0.33 0.38 0.32 – –
Few-shot 0.24 0.46 0.64 0.34 0.39 0.33 – –
RAG 0.23 0.45 0.63 0.34 0.39 0.33 – –
GraphRAG 0.25 0.47 0.65 0.35 0.40 0.34 – –
rEDMRec
(ours)0.26∗0.50 0.67 0.37 0.42 0.35 +4.0∗<0.001∗
Mixtral
8x7BZero-shot 0.24 0.47 0.66 0.35 0.40 0.35 – –
Few-shot 0.26 0.49 0.68 0.37 0.42 0.36 – –
RAG 0.25 0.48 0.67 0.36 0.41 0.35 – –
GraphRAG 0.27 0.50 0.69 0.38 0.42 0.37 – –
rEDMRec
(ours)0.28∗0.52 0.71 0.39 0.44 0.38 +3.7∗<0.001∗
Qwen3-14B Zero-shot 0.26 0.50 0.69 0.38 0.43 0.37 – –
Few-shot 0.28 0.52 0.71 0.40 0.45 0.39 – –
RAG 0.27 0.51 0.70 0.39 0.43 0.38 – –
GraphRAG 0.29 0.53 0.72 0.40 0.45 0.40 – –
rEDMRec
(ours)0.30∗0.55 0.73 0.42 0.47 0.41 +3.4∗<0.001∗
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 17 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 12 – continued
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
DeepSeek-R1
Distill-Qwen-14BZero-shot 0.28 0.52 0.71 0.40 0.45 0.40 – –
Few-shot 0.30 0.54 0.73 0.42 0.47 0.41 – –
RAG 0.29 0.53 0.72 0.41 0.46 0.40 – –
GraphRAG 0.31 0.55 0.74 0.43 0.48 0.42– –
rEDMRec
(ours)0.32∗0.57 0.75 0.44 0.49 0.42 +3.2∗<0.001∗
Phi-4 Zero-shot 0.25 0.48 0.68 0.36 0.42 0.36 – –
Few-shot 0.27 0.52 0.73 0.39 0.45 0.39 – –
RAG 0.26 0.50 0.71 0.38 0.44 0.37 – –
GraphRAG 0.28 0.54 0.76 0.41 0.47 0.40 – –
rEDMRec
(ours)0.29∗0.56 0.79 0.42 0.49 0.42 +3.6∗<0.001∗
Llama 4
ScoutZero-shot 0.27 0.51 0.70 0.39 0.44 0.38 – –
Few-shot 0.29 0.55 0.75 0.42 0.47 0.41 – –
RAG 0.28 0.53 0.73 0.40 0.46 0.39 – –
GraphRAG 0.30 0.57 0.78 0.43 0.49 0.42 – –
rEDMRec
(ours)0.31∗0.59 0.80 0.45 0.51 0.43 +3.3∗<0.001∗
GPT OSS
20BZero-shot 0.27 0.50 0.70 0.39 0.44 0.38 – –
Few-shot 0.29 0.54 0.75 0.41 0.47 0.40 – –
RAG 0.28 0.52 0.73 0.40 0.46 0.39 – –
GraphRAG0.30 0.56 0.78 0.43 0.49 0.42– –
rEDMRec
(ours)0.29 0.54 0.75 0.41 0.47 0.40 -3.3<0.001
B. Full Amazon Beauty Results
Table 13 reports the complete Methods×Models matrix on Amazon Beauty underlying the summary in Table 5
(Section 5.1).
Table 13: Complete Methods×Models results on Amazon Beauty (full held-out test split, 20 candidates/sample, seed
42). Impv (%) is the relative HR@1 gain of rEDMRec over thesecond-bestbaseline on the same student,Impv =
(Ours−SecondBest)∕SecondBest×100,followingRDRec(Wangetal.,2024b).Bestinbold,second-bestunderlined .𝑝isthe
exactMcNemar𝑝-valueforrEDMRecHR@1vs.thesecond-bestbaselineonthesamestudent(fullheld-out𝑛ML−1M=49893,
𝑛Beauty=1460,𝑛Steam=1460; approximate contingency from the table HR@1 rates).∗marks𝑝<0.05with rEDMRec ahead.
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Qwen2.5 3B Zero-shot 0.092 0.268 0.450 0.176 0.218 0.180 – –
Few-shot 0.104 0.280 0.450 0.182 0.224 0.192 – –
RAG 0.098 0.274 0.450 0.176 0.218 0.186 – –
GraphRAG 0.110 0.286 0.450 0.188 0.230 0.192 – –
rEDMRec
(ours)0.136∗0.329 0.479 0.220 0.262 0.224 +23.6∗0.037∗
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 18 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 13 – continued
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Llama 3.1 8B Zero-shot 0.0620.225 0.4500.128 0.180 0.162– –
Few-shot 0.0680.225 0.4500.128 0.180 0.162– –
RAG 0.0680.225 0.4500.128 0.180 0.162– –
GraphRAG0.074 0.225 0.4500.128 0.180 0.162– –
rEDMRec
(ours)0.071 0.225 0.450 0.137 0.180 0.162 -4.10.830
Gemma-4-
12BZero-shot 0.140 0.340 0.548 0.236 0.290 0.252 – –
Few-shot 0.152 0.352 0.560 0.248 0.302 0.258 – –
RAG 0.146 0.346 0.554 0.242 0.296 0.252 – –
GraphRAG 0.158 0.358 0.566 0.254 0.308 0.264 – –
rEDMRec
(ours)0.178 0.395 0.595 0.274 0.328 0.290 +12.70.166
Minimax
M2.5Zero-shot 0.152 0.364 0.572 0.254 0.308 0.264 – –
Few-shot 0.164 0.376 0.584 0.260 0.314 0.270 – –
RAG 0.158 0.370 0.578 0.260 0.314 0.270 – –
GraphRAG 0.170 0.382 0.590 0.266 0.320 0.276 – –
rEDMRec
(ours)0.187 0.416 0.616 0.289 0.343 0.290 +10.00.247
Mixtral
8x7BZero-shot 0.164 0.382 0.596 0.266 0.320 0.282 – –
Few-shot 0.176 0.394 0.608 0.278 0.332 0.288 – –
RAG 0.170 0.388 0.602 0.272 0.326 0.282 – –
GraphRAG 0.182 0.400 0.614 0.284 0.332 0.294 – –
rEDMRec
(ours)0.202 0.429 0.643 0.304 0.358 0.311 +11.00.188
Qwen3-14B Zero-shot 0.176 0.400 0.614 0.284 0.338 0.294 – –
Few-shot 0.188 0.412 0.626 0.296 0.350 0.306 – –
RAG 0.182 0.406 0.620 0.290 0.338 0.300 – –
GraphRAG 0.194 0.418 0.632 0.296 0.350 0.312 – –
rEDMRec
(ours)0.211 0.444 0.649 0.319 0.373 0.329 +8.80.269
DeepSeek-R1
Distill-Qwen-14BZero-shot 0.188 0.412 0.626 0.296 0.350 0.312 – –
Few-shot 0.200 0.424 0.638 0.308 0.362 0.318 – –
RAG 0.194 0.418 0.632 0.302 0.356 0.312 – –
GraphRAG 0.206 0.430 0.644 0.314 0.368 0.324 – –
rEDMRec
(ours)0.223 0.456 0.661 0.331 0.385 0.329 +8.30.280
Phi-4 Zero-shot 0.170 0.388 0.608 0.272 0.332 0.288 – –
Few-shot 0.182 0.412 0.638 0.290 0.350 0.306 – –
RAG 0.176 0.400 0.626 0.284 0.344 0.294 – –
GraphRAG 0.188 0.424 0.656 0.302 0.362 0.312 – –
rEDMRec
(ours)0.205 0.458 0.704 0.324 0.393 0.340 +9.00.264
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 19 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 13 – continued
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Llama 4
ScoutZero-shot 0.182 0.406 0.620 0.290 0.344 0.300 – –
Few-shot 0.194 0.430 0.650 0.308 0.362 0.318 – –
RAG 0.188 0.418 0.638 0.296 0.356 0.306 – –
GraphRAG 0.200 0.442 0.668 0.314 0.374 0.324 – –
rEDMRec
(ours)0.217 0.476 0.707 0.342 0.405 0.344 +8.50.275
GPT OSS
20BZero-shot 0.182 0.400 0.620 0.290 0.344 0.300 – –
Few-shot 0.194 0.424 0.650 0.302 0.362 0.312 – –
RAG 0.188 0.412 0.638 0.296 0.356 0.306 – –
GraphRAG0.200 0.436 0.668 0.314 0.374 0.324– –
rEDMRec
(ours)0.199 0.435 0.664 0.307 0.370 0.317 -0.51.000
C. Full Steam Results
Table 14 reports the complete Methods×Models matrix on Steam underlying the summary in Table 5 (Section 5.1).
Table 14: Complete Methods×Models results on Steam (full held-out test split, 20 candidates/sample, seed 42). Impv
(%) is the relative HR@1 gain of rEDMRec over thesecond-bestbaseline on the same student,Impv = (Ours −
SecondBest)∕SecondBest×100,followingRDRec(Wangetal.,2024b).Bestinbold,second-bestunderlined .𝑝istheexact
McNemar𝑝-value for rEDMRec HR@1 vs. the second-best baseline on the same student (full held-out𝑛ML−1M=49893,
𝑛Beauty=1460,𝑛Steam=1460; approximate contingency from the table HR@1 rates).∗marks𝑝<0.05with rEDMRec ahead.
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Qwen2.5 3B Zero-shot 0.106 0.274 0.450 0.188 0.224 0.180 – –
Few-shot 0.122 0.290 0.450 0.196 0.232 0.196 – –
RAG 0.114 0.282 0.450 0.188 0.224 0.188 – –
GraphRAG 0.130 0.298 0.450 0.204 0.240 0.196 – –
rEDMRec
(ours)0.158∗0.345 0.466 0.240 0.276 0.232 +21.5∗0.035∗
Llama 3.1 8B Zero-shot 0.0660.225 0.4500.126 0.180 0.162– –
Few-shot 0.0740.225 0.4500.126 0.180 0.162– –
RAG 0.0740.225 0.4500.126 0.180 0.162– –
GraphRAG0.082 0.225 0.4500.126 0.180 0.162– –
rEDMRec
(ours)0.076 0.225 0.450 0.133 0.180 0.162 -7.30.584
Gemma-4-
12BZero-shot 0.170 0.370 0.564 0.268 0.320 0.276 – –
Few-shot 0.186 0.386 0.580 0.284 0.336 0.284 – –
RAG 0.178 0.378 0.572 0.276 0.328 0.276 – –
GraphRAG 0.194 0.394 0.588 0.292 0.344 0.292 – –
rEDMRec
(ours)0.208 0.428 0.612 0.306 0.358 0.314 +7.20.356
Minimax
M2.5Zero-shot 0.186 0.402 0.596 0.292 0.344 0.292 – –
Few-shot 0.202 0.418 0.612 0.300 0.352 0.300 – –
RAG 0.194 0.410 0.604 0.300 0.352 0.300 – –
GraphRAG 0.210 0.426 0.620 0.308 0.360 0.308 – –
rEDMRec
(ours)0.224 0.460 0.644 0.330 0.382 0.321 +6.70.394
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 20 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 14 – continued
Model Method HR@1↑HR@5↑HR@10↑NDCG@5↑NDCG@10↑MRR↑Impv (%)𝑝
Mixtral
8x7BZero-shot 0.202 0.426 0.628 0.308 0.360 0.316 – –
Few-shot 0.218 0.442 0.644 0.324 0.376 0.324 – –
RAG 0.210 0.434 0.636 0.316 0.368 0.316 – –
GraphRAG 0.226 0.450 0.652 0.332 0.376 0.332 – –
rEDMRec
(ours)0.244 0.478 0.680 0.350 0.402 0.349 +8.00.276
Qwen3-14B Zero-shot 0.218 0.450 0.652 0.332 0.384 0.332 – –
Few-shot 0.234 0.466 0.668 0.348 0.400 0.348 – –
RAG 0.226 0.458 0.660 0.340 0.384 0.340 – –
GraphRAG 0.242 0.474 0.676 0.348 0.400 0.356 – –
rEDMRec
(ours)0.256 0.498 0.690 0.370 0.422 0.370 +5.80.392
DeepSeek-R1
Distill-Qwen-14BZero-shot 0.234 0.466 0.668 0.348 0.400 0.356 – –
Few-shot 0.250 0.482 0.684 0.364 0.416 0.364 – –
RAG 0.242 0.474 0.676 0.356 0.408 0.356 – –
GraphRAG 0.258 0.490 0.692 0.372 0.424 0.372 – –
rEDMRec
(ours)0.272 0.514 0.706 0.386 0.438 0.375 +5.40.426
Phi-4 Zero-shot 0.210 0.434 0.644 0.316 0.376 0.324 – –
Few-shot 0.226 0.466 0.684 0.340 0.400 0.348 – –
RAG 0.218 0.450 0.668 0.332 0.392 0.332 – –
GraphRAG 0.234 0.482 0.708 0.356 0.416 0.356 – –
rEDMRec
(ours)0.248 0.511 0.750 0.374 0.443 0.382 +6.00.411
Llama 4
ScoutZero-shot 0.226 0.458 0.660 0.340 0.392 0.340 – –
Few-shot 0.242 0.490 0.700 0.364 0.416 0.364 – –
RAG 0.234 0.474 0.684 0.348 0.408 0.348 – –
GraphRAG 0.250 0.506 0.724 0.372 0.432 0.372 – –
rEDMRec
(ours)0.264 0.535 0.756 0.398 0.459 0.388 +5.60.421
GPT OSS
20BZero-shot 0.226 0.450 0.660 0.340 0.392 0.340 – –
Few-shot 0.242 0.482 0.700 0.356 0.416 0.356 – –
RAG 0.234 0.466 0.684 0.348 0.408 0.348 – –
GraphRAG0.250 0.498 0.724 0.372 0.432 0.372– –
rEDMRec
(ours)0.245 0.488 0.708 0.359 0.421 0.359 -2.00.797
D. Experimental Settings
Table15liststhedefaulthyperparametersusedforallreportedruns.Unlessasubsectionstatesotherwise,every(model,
method,dataset)cellisevaluatedonthefullheld-outtestsplitwith20candidatespersample(1positive+19negatives)and
candidate-sampling seed42. The expanded implementation stack and full knob table appear in Appendix E.
E. Implementation Details and Hyperparameters
ThisappendixexpandsTable15withtheconcreteknobsinconfig.py(singlesourceoftruthforallreportedruns).Values
below are the repository defaults unless a subsection states otherwise.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 21 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 15
Default experimental settings
Component Setting Default
Data min. interactions / positive threshold ML-1M:20/3.5; Beauty & Steam:5/3.5(Steam:
0.0)
history / short-term window10/5items
teacher input history𝑘5latest train rows/user
negatives / seed19/42
Encoder model / dimall-MiniLM-L6-v2/384
Memory retrieval𝑚per channel5(Vector database)
channelslt,st,ip,cf
Teacher default modelgpt-5.4-mini
preference batch / overlap4/2
Controller max library size5000
ops Add / Delete / Modify / Keep
Debate optimize agents𝑘/ rounds3/1(sweep𝑘=1..10in Sec. 5.4)
max experiences/case6
Student protocol frozen pretrained LLM
Evaluation metrics HR@{1,5,10}, NDCG@{5,10}, MRR
evaluation split full held-out test (“all” samples)
Table 16
Expanded hyperparameters fromconfig.py(Appendix E).
Module Knob Default
Data / candidates
dataset registry ml-1m / amazon-beauty / steam
min interactions ML-1M: 20; Beauty/Steam: 5
positive threshold ML-1M/Beauty: 3.5; Steam: 0.0
history / short-term window 10 / 5 items
teacher input history𝑘5 latest train rows/user
negatives / split ratios / seed 19 / val=0.1, test=0.1 / 42
Encoder / memory
embedding model / dimall-MiniLM-L6-v2/ 384
max seq length / batch / normalize 256 / 64 / True
FAISS index / top-𝑚per channelFlatIP/ 5
channels long_term_preference, short_term_context,
item_perception, counterfactual
Teacher
default model / reasoning effortgpt-5.4-mini/ medium
max completion tokens / retries 8192 / 3
preference batch / overlap 4 / 2
extraction passes user_preference, item_perception_context,
item_perception_reasoning, counterfactual
Controller / debate optimize
ops / max library size / batch Add / Delete / Modify / Keep / 5000 / 32
debate agents𝑘/ rounds 3 / 1
max experiences/case 6
debate / arbiter temperature omit (API default)
Student / evaluation
protocol frozen pretrained LLM; memory toggles on by default
default local checkpoint nameQwen/Qwen2.5-3B-Instruct
max seq length 2048
metrics @𝑘HR@[1, 3, 5, 10], NDCG@5,10, MRR
eval samples 0 (0= full held-out test)
Implementation stack.Teacher / controller / debate calls use an OpenAI-compatible chat API (LLMConfig); the student
is afrozenpretrained LLM that ranks by retrieving from the experience bank. Dense retrieval uses FAISSFlatIPover all-
MiniLM-L6-v2 embeddings (𝑑=384). Counterfactual edges are stored in Neo4j (hybrid vector–graph channel). Candidate
sets are built offline (1 positive + 19 negatives; seed 42).
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 22 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 17
Debate optimization vs.𝑘-EPOCH on ML-1M (full held-out test; 20 candidates/sample; seed 42). Full debate uses𝑘=3. Control
= no-debate paraphrase (Mixtral). Dup.% / Reward / Spec. are bank-level signals.
𝑘-EPOCH HR@1 Mixtral HR@1 Qwen2.5 3B HR@1 Control Dup.%↓Reward↑Spec.↑
0 0.250 0.156 0.250 18.0 0.520 0.480
2 0.271 0.165 0.254 12.5 0.712 0.665
3 0.275 0.167 0.256 11.5 0.745 0.700
6 0.279 0.169 0.257 10.6 0.775 0.734
Δ(0→6)+0.029 +0.013 +0.007 −7.4 +0.255 +0.254
Table 18
Controller ablation at the end of the𝑘-EPOCH study (epoch 6, Mixtral 8x7B student). Full debate vs. no-debate control vs.
post-extraction bank (epoch 0).
Variant HR@1↑Dup.%↓Spec.↑
Post-extraction (epoch 0) 0.250 18.0 0.480
No-debate paraphrase 0.257 – –
Full debate(𝑘=3, 6 epochs)0.279 10.6 0.734
Full−Control+0.022– –
Table 19
Selected points from the number-of-agents sweep (Mixtral 8x7B, one epoch; full table in Table 9). Calls/case=𝑘⋅𝑛𝑟+1arbiter.
𝑘HR@1↑Spec.↑Dup.%↓Calls/case
1 0.255 0.520 16.0 2
3 0.273 0.663 11.9 4
4 (𝑘∗)0.277 0.699 11.0 5
10 0.283 0.756 9.6 11
Reproducibilitynotes.AllRQ1cellsusethechronologicaltrain/validation/testsplitwith20candidates/sampleandseed42.
Channelablationsflipthefouruse_*_memoryflagsinStudentConfig.DebatesweepsvaryOptimizeKnowledgeConfig.debate_agents_k
and the number of𝑘-EPOCHs while holding the student frozen.
F. Debate and Controller Ablations
Thisappendixtabulatesthecontroller/debatevariantsthatsupportRQ4(Section5.4).Thedefaultconfigurationis𝑘=3
debate agents,𝑛𝑟=1round per epoch, and a single LLM arbiter (Appendix E). We report (i)𝑘-EPOCH trajectories with a
no-debate paraphrase control and (ii) a one-epoch sweep over the number of agents.
Variant definitions.Full debate:𝑘-agent critique + arbiter + controller Add/Delete/Modify/Keep commits.No-debate
paraphrase:refreshes entry wording without the critique-and-revise loop (control in Figure 6).Single-agent (𝑘=1):one
persona + arbiter (no multi-agent disagreement); used as the𝑘=1anchor in Table 9.
Table 17 shows that most of the Mixtral gain arrives by epoch 2–3, after which returns diminish; the paraphrase control
staysnearlyflat(+0.007HR@1),isolatingthedebateloop.Table18summarizestheend-statecontrollerablation.Theagent-
countsweep(main-textTable9)placesthequality-per-costkneeat𝑘∗=4;beyondthat,HR@1gainsare<0.01forsixextra
LLM calls per case.
G. Additional Ablation Figures
This appendix collects ablation visuals that support Section 5.2 but were omitted from the main text for space. Figure 9
repeatsthecross-modelchannelablationunderMRR;Figure10showsthefullnumericablationmatrix;Figure11aggregates
channel importance; and Figure 12 breaksΔHR@1 down per ablation-panel backbone.
H. Dataset Statistics
Table 20 reports user, item, and interaction counts after the𝑘-core filter used in Section 4.1 (𝑘=20on ML-1M;𝑘=5
on Amazon Beauty and Steam), together with feedback type, positive threshold, density, and the number of chronological
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 23 of 25

rEDMRec: Reasoning Distillation into Experience Memory
w/o Short-Term w/o Long-Term w/o Item-Perc.w/o Counterfactualw/o Memory
Channel removedGPT-5-mini
GPT-5.4-mini
GPT-OSS-120B
Qwen3-32B
Minimax M2.5
Llama-3.3-70B
Llama-3.1-8BStudent model-0.06 -0.06 -0.07 -0.05 -0.10
-0.05 -0.05 -0.06 -0.05 -0.08
-0.01 -0.01 -0.01 -0.01 -0.02
-0.05 -0.05 -0.06 -0.05 -0.08
-0.05 -0.05 -0.06 -0.05 -0.08
-0.02 -0.02 -0.02 -0.02 -0.04
-0.02 -0.02 -0.02 -0.02 -0.04Memory channel ablation: ΔMRR across student models
-0.10-0.050.000.050.10
ΔMRR
Figure 9:Channel ablation heatmap under MRR (ΔMRR vs. full memory), complementary to Figure 4.
Table 20
Dataset statistics after𝑘-core filtering (Section 4.1). Density=|𝑅|∕(|𝑈|⋅|𝐼|). Test𝑛is the number of 20-candidate ranking
samples in the chronological held-out split.
Dataset Domain Feedback #Users #Items #Inter. Dens. Sparsity𝑘-core Pos. thr. Test𝑛
ML-1M Movies Explicit (1–5) 6,040 3,706 1,000,209 4.47% Very Low 20𝑟>3.549,893
Amazon Beauty Beauty products Explicit (1–5) 1,620 7,116 14,984 0.13% Very High 5𝑟>3.51,460
Steam Games Implicit (play) 62,936 10,978 5,077,150 0.73% Medium 50.0(all logged) 1,460
held-out ranking samples (Test𝑛) used for RQ1. Density is|𝑅|∕(|𝑈|⋅|𝐼|). ML-1M is dense with explicit ratings; Beauty
is extremely sparse after 5-core filtering onAll_Beauty; Steam is implicit (owned/played games treated as positives). Split
construction follows a chronological leave-suffix protocol (train / validation / test ratios0.8/0.1/0.1per user sequence),
with 20-candidate ranking samples (1 positive + 19 negatives, seed42).
I. Prompts and Outputs
This appendix documents the teacher extraction prompts and the distilled experience-memory outputs that the student
retrieves at ranking time (Sections 3.5–3.6.1). Following the presentation style of ReasoningRec (Bismay et al., 2025),
we highlight semantically distinct spans with color: role assignment, long-/short-term preference inputs, CoT instructions,
guardrails, and structured JSON fields in the prompts (Table 21); liked attributes, dislikes, item-perception rationale, and
counterfactual anchor/contrast/condition fragments in the one-line per-channel outputs (Table 22).
Colorlegend.Role,long-termpreference,short-termcontext,CoT/compareinstruction,guardrails,structuredoutputfields,
liked attributes, dislikes, item-perception rationale, anchor preferred, contrast rejected, counterfactual condition.
J. Examples of rEDMRec-generated Predictions
WeillustratethefullrEDMRecrankingcallforoneusereachonAmazonBeautyandSteam(Tables23–24).Eachexample
showsthefrozenstudent’sinput–chronologicalhistory𝐻𝑢,thecandidateset𝐶𝑢(abbreviated),andthetopretrievedentries
fromthefourteacherextractionchannels(Preferencepref,Contextctx,Reasoningreas,Counterfactualcf)–followedbythe
student’s ranked list and a short rationale. Item titles are taken from the public catalogs; channel texts follow the distillation
schemainSection3.6.1.Colorhighlightingmarkslikedvs.dislikedhistoryitems,theheld-outtargetamongcandidates,each
extraction channel, and the top of the ranked output (same palette as Appendix I).
Colorlegend(sharedwithAppendixI).Historyblock,candidateset,retrieved-channelheader,target/rankedoutput,liked
/high-engagement,disliked/avoided,Preferenceextraction(pref),Contextextraction(ctx),Reasoningextraction(reas),cf
anchor, cf contrast, Counterfactual extraction (cf).
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 24 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Student Variant HR@1 HR@3 HR@10 NDCG@3 NDCG@10 MRR ΔHR@1 ΔMRR
GPT-5-mini Full 0.40 0.60 0.80 0.505 0.601 0.539 +0.00 +0.00
w/o Short-Term 0.338 0.507 0.816 0.425 0.556 0.478 -0.06 -0.06
w/o Long-Term 0.34 0.51 0.815 0.428 0.557 0.48 -0.06 -0.06
w/o Item-Perc. 0.325 0.487 0.819 0.409 0.546 0.466 -0.07 -0.07
w/o Counterfactual 0.345 0.518 0.814 0.434 0.561 0.485 -0.05 -0.05
w/o Memory 0.30 0.45 0.825 0.377 0.528 0.441 -0.10 -0.10
GPT-5.4-mini Full 0.31 0.49 0.79 0.405 0.535 0.455 +0.00 +0.00
w/o Short-Term 0.255 0.408 0.804 0.336 0.495 0.402 -0.05 -0.05
w/o Long-Term 0.257 0.411 0.803 0.338 0.497 0.404 -0.05 -0.05
w/o Item-Perc. 0.244 0.391 0.806 0.321 0.487 0.391 -0.07 -0.06
w/o Counterfactual 0.262 0.417 0.802 0.344 0.50 0.408 -0.05 -0.05
w/o Memory 0.23 0.37 0.81 0.303 0.477 0.377 -0.08 -0.08
Minimax M2.5 Full 0.35 0.51 0.76 0.42 0.535 0.455 +0.00 +0.00
w/o Short-Term 0.295 0.428 0.774 0.35 0.495 0.402 -0.05 -0.05
w/o Long-Term 0.297 0.431 0.773 0.352 0.497 0.403 -0.05 -0.05
w/o Item-Perc. 0.284 0.411 0.776 0.336 0.487 0.39 -0.07 -0.06
w/o Counterfactual 0.302 0.438 0.772 0.358 0.50 0.408 -0.05 -0.05
w/o Memory 0.27 0.39 0.78 0.318 0.477 0.377 -0.08 -0.08
Qwen3-32B Full 0.34 0.50 0.75 0.412 0.524 0.445 +0.00 +0.00
w/o Short-Term 0.285 0.418 0.764 0.342 0.484 0.392 -0.05 -0.05
w/o Long-Term 0.287 0.421 0.763 0.345 0.486 0.393 -0.05 -0.05
w/o Item-Perc. 0.274 0.401 0.766 0.328 0.476 0.38 -0.07 -0.06
w/o Counterfactual 0.292 0.427 0.762 0.35 0.489 0.398 -0.05 -0.05
w/o Memory 0.26 0.38 0.77 0.31 0.466 0.367 -0.08 -0.08
GPT-OSS-120B Full 0.30 0.47 0.71 0.406 0.499 0.43 +0.00 +0.00
w/o Short-Term 0.293 0.459 0.712 0.397 0.493 0.423 -0.01 -0.01
w/o Long-Term 0.293 0.459 0.712 0.397 0.493 0.423 -0.01 -0.01
w/o Item-Perc. 0.291 0.457 0.712 0.395 0.492 0.422 -0.01 -0.01
w/o Counterfactual 0.293 0.46 0.712 0.398 0.494 0.424 -0.01 -0.01
w/o Memory 0.28 0.44 0.715 0.381 0.484 0.411 -0.02 -0.02
Llama-3.1-8B Full 0.11 0.18 0.27 0.131 0.175 0.159 +0.00 +0.00
w/o Short-Term 0.091 0.152 0.275 0.107 0.161 0.141 -0.02 -0.02
w/o Long-Term 0.092 0.153 0.275 0.107 0.162 0.141 -0.02 -0.02
w/o Item-Perc. 0.087 0.146 0.276 0.102 0.158 0.137 -0.02 -0.02
w/o Counterfactual 0.093 0.155 0.274 0.109 0.163 0.142 -0.02 -0.02
w/o Memory 0.07 0.12 0.28 0.079 0.146 0.119 -0.04 -0.04
Llama-3.3-70B Full 0.10 0.23 0.34 0.16 0.213 0.177 +0.00 +0.00
w/o Short-Term 0.081 0.202 0.345 0.137 0.199 0.159 -0.02 -0.02
w/o Long-Term 0.082 0.203 0.344 0.137 0.20 0.16 -0.02 -0.02
w/o Item-Perc. 0.077 0.196 0.346 0.132 0.196 0.155 -0.02 -0.02
w/o Counterfactual 0.084 0.205 0.344 0.139 0.201 0.161 -0.02 -0.02
w/o Memory 0.06 0.17 0.35 0.109 0.183 0.138 -0.04 -0.04Complete ablation results (appendix)
Bold Full rows. Green/red: positive/negative Δ.
Figure 10:Complete channel-ablation table (absolute metrics and deltas) across the seven ablation-panel backbones.
K. Failure Cases and Qualitative Memory Edits
This appendix complements the positive qualitative cases in Section 5.5 and the end-to-end traces in Appendix J with
failuremodesandborderlineedits:backbonesthatlosetoGraphRAG,controllereditsthatrewritetastetooaggressively,and
the short-term compression pattern that lowers lexical specificity while remaining actionable.
H.1 Ranking failures vs. GraphRAG.On ML-1M, rEDMRec trails GraphRAG on Llama 3.1 8B (Impv= −11.1%)
and GPT OSS 20B (Impv= −3.3%; Appendix A). We attribute the Llama failure to weak instruction-following on the
concatenated memory prompt (Section 6), not to an empty bank: the same bank yields positive Impv on stronger students.
On Beauty/Steam the same two backbones again show near-zero or negative Impv (Appendix B–C), so the limitation is
backbone-dependent rather than dataset-specific.
H.2 Qualitative edits: success vs. risk.Table 26 contrasts a beneficial long-term rewrite (Case S1; also Case A in the
maintext),ataste-flipriskwheredebateoverwritesanearliersci-fiprofilewithnoir/crime(CaseR1),short-termcompression
(Case C1), and a strongly item-grounded item-perception fix (Case S2).
H.3 Capacity-dependent channel reversals.Channel ablations (Section 5.2) show that removing long-term, item-
perception,orcounterfactualmemorycanimproveHR@1onthestrongestablation-panelstudent(gpt-5-mini),i.e.thebank
caninjectnoisewhenthebackbonealreadyrankswellfromcandidatesalone.Short-termcontextremainstheonlyconsistently
beneficial channel across tiers — a practical failure mode for “always retrieve all four channels” deployments on saturated
students.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 25 of 25

rEDMRec: Reasoning Distillation into Experience Memory
-0.1 0.0 0.1
mean ΔHR@1w/o Short-Termw/o Long-Termw/o Item-Perc.w/o Counterfactualw/o Memory
-0.062-0.060-0.075-0.055-0.100Strong (GPT-5-mini)
-0.1 0.0 0.1
mean ΔHR@1-0.055-0.053-0.066-0.048-0.080Mid-cap
-0.025 0.000 0.025
mean ΔHR@1-0.007-0.007-0.009-0.007-0.020Saturated (120B)
-0.05 0.00 0.05
mean ΔHR@1-0.019-0.018-0.023-0.017-0.040Weak (Groq 8B/70B)Channel ablation by student capacity tier
Figure 11:Aggregated channel-importance summary used to derive Table 6.
Table 21
Teacher extraction prompts for the four passes𝑝∈(Section 3.5). Colored spans mark the role, inputs, CoT instruction, and
JSON schema. Full templates live inteacher/prompt_templates.py.
Pass / channel Prompt (abbreviated)
pref→lt,stYou are an expert movie recommendation analyst implementing User Preference Maintenance. Simulate recurrent
updates over the interaction sequence (oldest→newest) in overlapping batches. Integrate each batch into an updated
long-term preference state and record the last-batch short-term interest. List consistent dislikes / avoided tones.
Prefer concrete movie-relevant language; avoid empty platitudes. Output STRICT JSON only (no markdown). Fields:
maintenance_trace[], long_term_preferences, short_term_preferences, dislikes, reasoning.
ctx/reas→ipYou are implementing Item Perception Analysis / recommendation reasoning. For each history item and candidate,
produce (1) objective factual description, (2) first-person [Comment:] as this user, (3) candidate key phrases; then a
five-step CoT matching themes→attributes→candidates→compare→recommend. Condition on long-term preferences
and short-term focus. Use exact title strings as JSON keys; include every history and candidate title. Fields:
user_history_perception, candidate_perception, steps[1..5], recommended_item, reasoning_summary.
cf→cfYou are a contrastive reasoning analyst for movie recommendations. Given preferences, an anchor (chosen) item,
and a contrast (hard-negative) item: (1) why the anchor is preferred; (2) why the contrast is unsuitable; (3)
a hypothetical condition under which the contrast would outrank the anchor; (4) robustness. Do not invent
facts absent from the provided preference and item text. Output STRICT JSON only. Fields: anchor_item,
contrast_item, why_anchor_preferred, why_contrast_rejected, counterfactual_condition, counterfactual_outcome,
robustness, rationale.
Table 22
Example distilled memory outputs (one line per channel) from the persisted bank for user 2 on ML-1M. Each line is the committed
entry text after distillation (Section 3.6.1); colored spans highlight the ranking-relevant fragments.
Channel Distilled experience (one line)
ltHe strongly prefers character-driven dramas with emotional depth, mature themes, moral conflict, and strong
performances. Repeated high ratings cluster around courtroom/drama (A Few Good Men), inspirational sports/drama
... Dislikes: He consistently rates lower when drama is diluted by broad, quirky, or eccentric comedy tones. Nurse Betty
is the clearest dislike (1.0), wh...
stIn the most recent items, interest appears to tilt further toward intimate, human-centered drama and reflective sci-fi, with
high ratings for Driving Miss Daisy and Close Encounters. At the same time, gritty crime/action titles and war-related
films have been less successful rece...
ipThe user’s history strongly favors light, charming comedies with romance, warmth, and quirky optimism, especially films
like Shakespeare in Love, Strictly Ballroom, Shall We Dance?, Groundhog Day, and Forrest Gump. Among the candidates,
For Love or Money is the closest match because it sits in the r...
cfAnchor (One Flew Over the Cuckoo’s Nest): One Flew Over the Cuckoo’s Nest fits the user’s strongest pattern: serious,
award-caliberdramawithintensecharacterfocus,moralconflict,andweightythemes... Contrast(Dante’sPeak):Dante’s
Peak is primarily an action-thriller/disaster film, which is comparatively light on the kind of prestige, historical, or
biographica... If the user were instead seeking a tense, fast-paced disaster thriller for pure entertainment rather than
a prestige drama, then Dante’s Peak would rank higher because its volcanic-disaster suspense, clear genre pacing, and
spect...
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 26 of 25

rEDMRec: Reasoning Distillation into Experience Memory
-0.08 0.00 0.08
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.100
-0.075
-0.062
-0.060
-0.055(a) ΔHR@1
-0.08 0.00 0.08
Δ-0.100
-0.075
-0.062
-0.060
-0.055(b) ΔNDCG@1
-0.08 0.00 0.08
Δ-0.098
-0.074
-0.061
-0.059
-0.054(c) ΔMRRAblation — GPT-5-mini @ n=30,000
(a)gpt-5-mini(strong)
-0.10 -0.05 0.00 0.05 0.10
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.080
-0.066
-0.055
-0.053
-0.048(a) ΔHR@1
-0.10 -0.05 0.00 0.05 0.10
Δ-0.080
-0.066
-0.055
-0.053
-0.048(b) ΔNDCG@1
-0.05 0.00 0.05
Δ-0.078
-0.065
-0.054
-0.052
-0.047(c) ΔMRRAblation — GPT-5.4-mini @ n=30,000 (b)gpt-5.4-mini(mid)
-0.06 0.00 0.06
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.080
-0.066
-0.055
-0.053
-0.048(a) ΔHR@1
-0.06 0.00 0.06
Δ-0.080
-0.066
-0.055
-0.053
-0.048(b) ΔNDCG@1
-0.05 0.00 0.05
Δ-0.078
-0.065
-0.053
-0.052
-0.047(c) ΔMRRAblation — Qwen3-32B @ n=30,000
(c) Qwen3-32B (mid)
-0.015 0.000 0.015
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.020
-0.009
-0.007
-0.007
-0.007(a) ΔHR@1
-0.015 0.000 0.015
Δ-0.020
-0.009
-0.007
-0.007
-0.007(b) ΔNDCG@1
-0.015 0.000 0.015
Δ-0.020
-0.009
-0.007
-0.007
-0.007(c) ΔMRRAblation — GPT-OSS-120B @ n=30,000 (d) GPT-OSS-120B (saturated)
-0.03 0.00 0.03
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.040
-0.023
-0.019
-0.018
-0.017(a) ΔHR@1
-0.03 0.00 0.03
Δ-0.040
-0.023
-0.019
-0.018
-0.017(b) ΔNDCG@1
-0.025 0.000 0.025
Δ-0.039
-0.022
-0.018
-0.018
-0.016(c) ΔMRRAblation — Llama-3.3-70B @ n=30,000
(e) Llama 3.3 70B (weak)
-0.050 -0.025 0.000 0.025 0.050
Δw/o Memory
w/o Item-Perc.
w/o Short-Term
w/o Long-Term
w/o Counterfactual-0.040
-0.023
-0.019
-0.018
-0.017(a) ΔHR@1
-0.050 -0.025 0.000 0.025 0.050
Δ-0.040
-0.023
-0.019
-0.018
-0.017(b) ΔNDCG@1
-0.025 0.000 0.025
Δ-0.039
-0.022
-0.018
-0.018
-0.016(c) ΔMRRAblation — Llama-3.1-8B @ n=30,000 (f) Llama 3.1 8B (weak)
Figure 12:Per-backbone channel ablation (ΔHR@1 when removing each channel). Negative bars indicate a beneficial channel.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 27 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 23
Example of an rEDMRec prediction trace onAmazon Beauty(illustrative end-to-end I/O; item titles from the public catalog).
Colored spans mark likes/dislikes in history, the held-out target among candidates, the four retrieved extraction channels, and the
top-ranked output.
Field Content
User AG7W...BVDA
History𝐻𝑢 Liked: MyGift Soft Padded Spa Bath Pillow; Liked: Nice ’n Easy Permanent Color 9G Light Golden
Blonde; Liked: Avon Glimmersticks Waterproof Eyeliner (Smokey Grey); Liked: Yes To Sensitive Facial
CleansingWipes;Liked:LaClaireFoamingBotanicalFacialCleanser;Disliked:4DSilkFiberLashMascara
(clumpy / heavy); Liked: MOSTORY Glitter Crystal Liquid Eyeshadow Set; Liked: GLOW BOOSTER
SERUM; Liked: Oval Large Makeup Brushes (Rose Gold); Liked: The Vegan Glow Quinoa Protein
Shampoo Bar; Liked: Bloomeffects Natural Tulip Dew Face Cream
Candidates𝐶𝑢 JUNGSAEMMOOL Minifying Cica Mist Balm (target); 4D Silk Fiber Lash Mascara Black; Rhinestone
CrystalPaddedHeadband;SeaMagikPinkSaltConditioner;BIOSSANCEMarineAlgaeEyeCreamMini;
Foamie Shampoo Bar Hibiskiss; Intraceuticals Rejuvenate Eye Masks; BeautyStat Universal Moisture
Essence (Squalane)
Retrieved experience channels𝑅𝑘
Preference
extraction (pref)Prefers clean, botanical / cruelty-free skincare and soft everyday makeup tools; repeatedly high-rates
serums, cream cleansers, and gentle face care over heavy glam accessories. Dislikes: Avoids heavy, clumpy
mascara and overly decorative rhinestone accessories that read as party glam rather than daily care.
Context
extraction (ctx)Recent purchases emphasize glow serums, vegan shampoo bars, and non-toxic face cream – a short-term
tilt toward soothing, clean-beauty maintenance rather than color cosmetics.
Reasoning
extraction (reas)Cica Mist Balm matches the user’s soothing / clean-skincare lane (cica + mist balm for calming),
whereas rhinestone headbands and fiber mascara clash with recent botanical preferences.
Counterfactual
extraction (cf)Anchor (Cica Mist Balm): Fits the clean-beauty, calming-care pattern reinforced by recent Tulip Dew
cream and glow serum. Contrast (Rhinestone Crystal Headband): Statement bridal/party accessory;
decorative rather than skincare-functional. If the user were shopping for a one-off formal event accessory
instead of daily facial care, the contrast would rank higher.
Student output RANKED LIST: 1. JUNGSAEMMOOL Minifying Cica Mist Balm; 2. BeautyStat Universal Moisture
Essence (Squalane); 3. BIOSSANCE Marine Algae Eye Cream Mini; 4. Intraceuticals Rejuvenate Eye
Masks; 5. Foamie Shampoo Bar Hibiskiss; 6. Sea Magik Pink Salt Conditioner; 7. 4D Silk Fiber Lash
Mascara Black; 8. Rhinestone Crystal Padded Headband
Rationale: Top ranks stay in soothing skincare / moisture; glam mascara and rhinestone accessories are
demoted using short-term clean-beauty context and the counterfactual contrast.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 28 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 24
ExampleofanrEDMRecpredictiontraceonSteam(illustrativeend-to-endI/O;itemtitlesfromthepubliccatalog).Coloredspans
mark likes/dislikes in history, the held-out target among candidates, the four retrieved extraction channels, and the top-ranked
output.
Field Content
User 765611980946...
History𝐻𝑢 Played: Garry’s Mod (high playtime); Played: Half-Life 2; Played: Half-Life 2: Episode One; Played:
Portal; Played: Portal 2; Played: The Binding of Isaac; Played: PlanetSide 2; Low play: Dota 2 Test
Candidates𝐶𝑢 Half-Life 2: Episode Two (target); Counter-Strike: Global Offensive; PAYDAY 2; Warframe; Terraria;
Left 4 Dead 2; The Expendabros; Yosumin!
Retrieved experience channels𝑅𝑘
Preference
extraction (pref)Strong Valve narrative / puzzle-FPS taste: Half-Life 2 saga and Portal series dominate, with sandbox
creativity (Garry’s Mod) and occasional indie rogue-likes (Isaac). Dislikes: Little engagement with pure
MOBA test clients; competitive live-service shooters are secondary to story/puzzle FPS.
Context
extraction (ctx)Recent high-engagement cluster is Portal 2 + Half-Life episodes; short-term focus is completing the
Valve narrative loop rather than opening new live-service grinders.
Reasoning
extraction (reas)Episode Two is the direct narrative continuation of Episode One already in history; CS:GO / Warframe
offer multiplayer loops less aligned with the story-FPS preference.
Counterfactual
extraction (cf)Anchor (Half-Life 2: Episode Two): Continues the exact Half-Life 2 story the user already invested in.
Contrast (Warframe): Free-to-play grind / live-service loop; weak narrative continuity with Portal/HL2.
If the user wanted a long-horizon multiplayer grind instead of finishing a single-player story arc, the
contrast would rank higher.
Student output RANKED LIST: 1. Half-Life 2: Episode Two; 2. Left 4 Dead 2; 3. Terraria; 4. PAYDAY 2; 5. Counter-
Strike: Global Offensive; 6. Warframe; 7. The Expendabros; 8. Yosumin!
Rationale: Episode Two leads via long-term Valve narrative memory and item-perception continuity;
co-op FPS is secondary; mismatched casual / grind titles sink.
Table 25
Failure / borderline RQ1 cells (Impv vs. second-best; typically GraphRAG). Negative Impv = GraphRAG ahead.
Dataset Student Impv (%) Note
ML-1M Llama 3.1 8B−11.1GraphRAG best; weak IF
ML-1M GPT OSS 20B−3.3near-saturated student
Beauty Llama 3.1 8B−4.1not significant
Beauty GPT OSS 20B−0.5tie within noise
Steam Llama 3.1 8B−7.3GraphRAG best
Steam GPT OSS 20B−2.0GraphRAG best
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 29 of 25

rEDMRec: Reasoning Distillation into Experience Memory
Table 26
Qualitative memory edits from the persisted bank (experiments/bank_evolution_cases.json). S = success-like; R = risk / failure
mode; C = compression.
Case Spec. Before After Interpretation
S1/lt 0.214→
0.479“Very limited data: the only rated
film is a classic action-adventure,
suggesting a preference for high-
energy, heroic, escapist storytelling
with suspense and...”“Long-term: favors mainstream
1990s action buddy-cop films
– franchise sequels, star-driven
chemistry, high-energy action
with comedic interplay. Upweight
these...”Hedge -> ranking rule; +spec.
R1/lt 0.450→
0.625“Core taste is classic, lighthearted
sci-fi with ensembles and humor,
with some room for action SF. Hor-
ror is avoided, and the user responds
best to accessible,...”“User 17 strongly prefers classic
and neo-noir/crime prestige dramas.
Downweight short-term popularity
signals and upweight niche/indie,
1990s-era, and foreign-...”Taste flip risk: sci-fi -> noir;
may discard valid prior signal.
C1/st 0.300→
0.500“No recent viewing items were pro-
vided, so no emerging short-term
interests can be detected.”“Session: prioritize late-80s/90s Hol-
lywood action comedies with buddy
dynamics and franchise entries for
top slots; include at least one diverse
alternative pe...”Empty note -> session rule;
aggregate st spec. can drop.
S2/ip 0.350→
0.850“The strongest match was only
ranked 5th, so Hit@10 was good but
Hit@1/MRR suffered. For this user,
place the best Rocky-like inspira-
tional drama at rank 1 when...”“Rocky (1976): User 4 chose Rocky
over a higher-ranked 80s action ti-
tle, signaling a preference for 1970s
character-driven underdog sports
dramas. When ranking,...”Vague Hit@1 complaint ->
item-keyed boost.
M.H. Nguyen et al.:Preprint submitted to ElsevierPage 30 of 25