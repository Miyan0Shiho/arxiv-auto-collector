# Navigating Sparse Evidence: Agentic Visual RAG via Explicit Context Selection and Consolidation

**Authors**: Yucheng Shen, Lingyong Yan, Jiulong Wu, Shuaiqiang Wang, Jianmin WU, Dawei Yin, Min Cao

**Published**: 2026-09-14 16:11:35

**PDF URL**: [https://arxiv.org/pdf/2609.15800v1](https://arxiv.org/pdf/2609.15800v1)

## Abstract
Visual Retrieval-Augmented Generation (VRAG) empowers models to navigate and answer queries about visually rich documents by retrieving relevant page images as visual evidence and reasoning over their content. However, effectively utilizing this visual evidence is usually impeded by two main challenges. First, answer-relevant evidence is sparse and may be concentrated in a small region of one page or dispersed across multiple pages. Second, existing agentic methods often generate answers based on raw exploration trajectories or compressed textual memories rather than an explicitly organized set of supporting images, making answers susceptible to exploration noise and obscuring the evidence-backed reasoning trace. We argue that the bottleneck lies not only in evidence discovery but also in its preservation and organization before answer generation. We propose SCoRE (Selection and Consolidation for Robust Evidence), a unified agent loop for explicit evidence selection and consolidation. During exploration, SCoRE retains only query-relevant observations and their source pointers in a maintained textual ledger, preserving earlier evidence while keeping the visual context bounded. At termination, it reloads the referenced original images and consolidates the visual evidence for answering, arranging it into a logical sequence. This decouples final reasoning from exploratory trial-and-error while ensuring strict visual grounding via indexed claim-to-image linkages. To enable end-to-end optimization of this unified rollout, our training paradigm combines filtered cold-start trajectory distillation with evidence-aware reinforcement learning, whose reward promotes evidence coverage, consolidation compactness, and answer correctness.

## Full Text


<!-- PDF content starts -->

Navigating Sparse Evidence: Agentic Visual RAG via
Explicit Context Selection and Consolidation
Yucheng Shen1,2Lingyong Yan2∗, Jiulong Wu2, Shuaiqiang Wang,
Jianmin WU2, Dawei Yin2, Min Cao1∗
1School of Computer Science and Technology, Soochow University2Baidu Inc.
ycshensudaer@stu.suda.edu.cn, lingyongy@gmail.com, mcao@suda.edu.cn
Abstract
Visual Retrieval-Augmented Generation (VRAG) empowers
models to navigate and answer queries about visually rich
documents by retrieving relevant page images as visual evi-
dence and reasoning over their content. However, effectively
utilizing this visual evidence is usually impeded by two main
challenges. First, answer-relevant evidence is sparse and may
be concentrated in a small region of one page or dispersed
across multiple pages. Second, existing agentic methods of-
ten generate answers based on raw exploration trajectories or
compressed textual memories rather than an explicitly orga-
nized set of supporting images, making answers susceptible
to exploration noise and obscuring the evidence-backed rea-
soningtrace.Wearguethatthebottleneckliesnotonlyinev-
idencediscoverybutalsoinitspreservationandorganization
beforeanswergeneration.WeproposeSCORE(Selectionand
COnsolidationforRobustEvidence),aunifiedagentloopfor
explicit evidence selection and consolidation. During explo-
ration,SCOREretains only query-relevant observations and
their source pointers in a maintained textual ledger, preserv-
ingearlierevidencewhilekeepingthevisualcontextbounded.
At termination, it reloads the referenced original images and
consolidates the visual evidence for answering, arranging
it into a logical sequence. This decouples final reasoning
from exploratory trial-and-error while ensuring strict visual
grounding via indexed claim-to-image linkages. To enable
end-to-end optimization of this unified rollout, our training
paradigm combines filtered cold-start trajectory distillation
with evidence-aware reinforcement learning, whose reward
promotesevidencecoverage,consolidationcompactness,and
answer correctness. Experiments on three established VRAG
benchmarks demonstrate thatSCOREachieves state-of-the-
art overall accuracy on each benchmark across various back-
bone scales, with further gains from both training stages.
Introduction
Visual Retrieval-Augmented Generation (VRAG) (Wang
et al. 2026, 2025; Shen et al. 2026) extends traditional
retrieval-augmented generation (Arslan et al. 2024) to vi-
sually rich documents such as slides, reports, and scanned
PDFs, retrieving and reasoning over page images. In these
documents, evidential contentis encoded not only intextual
∗Corresponding authors.
Copyright©2027, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
 (a) Uneven  Evidence Distri bution  
...across
imageswithin
a image
sparse evidence with noise pages
tiny relevant
region
 (b) Implicit Evidence Organization  
...
turn 1
search
Answer
Generationturn 2
searchturn 3
bboxturn n
searchwithout
organization
⚠Figure 1: Illustration of two challenges in visual evidence
utilization in VRAG: (a) uneven evidence distribution; (b)
implicit evidence organization.
form but also through charts, tables, layout structures, and
spatial relationships. Recent advances in Vision-Language
Models (VLMs) (Bai et al. 2025a,b; Liu et al. 2024, 2023)
enableVRAGmethodslikeVRAG-RL(Wangetal.2026)to
leverage raw visual inputs, preserving critical layout, struc-
tural, and multimodal cues that are often lost or distorted by
traditional OCR-based methods (Zhang et al. 2025).
In common VRAG settings, rendered document pages
serve as the basic retrieval unit, resulting in a substan-
tially coarser granularity than passage-level text retrieval.
This page-level retrieval paradigm poses two practical chal-
lenges for effective visual evidence utilization. (1)Uneven
evidence distribution.Answer-relevant evidence
may be localized within a small region of a single page
or scattered across multiple different pages (Figure 1 (a)).
The agent must therefore determine which pages are rel-
evant, how many of them are needed, and which re-
gionsrequirecloserinspection.(2)Implicit evidence
organization.Evidence is often discovered incremen-
tallyacrossmultipleretrievalandinspectionsteps.However,
the discovered observations may remain organized accord-
ing to the exploration process itself rather than the logical
structure required for coherent answering (Figure 1 (b)).
Existing agentic methods (Wang et al. 2026; Shen et al.
2026) improve evidence discovery through iterative search
arXiv:2609.15800v1  [cs.AI]  14 Sep 2026

and region-level zoom-in. VRAG-RL retains visual obser-
vations in the interaction trajectory, whereas VISOR distills
themintoastructuredtextualevidencespace.However,nei-
ther explicitly reconstructs the original visual evidence into
a compact, logically ordered chain for final answer genera-
tion or establishes claim-level links to visual sources. These
limitationsmotivateourcentralinsight:explorationshould
discover potential evidence, while consolidation should
select, restore, and organize it into an answer-oriented
visual evidence chain for grounded generation.
Toaddressthisbottleneck,weproposeSCORE(Selection
andCOnsolidation forRobustEvidence), a unified agent
loop for evidence organization in VRAG (Figure 2). The
agentexploresvisualcontentthroughcoarseimageretrieval
and fine-grained bounding-box zoom-in, treating full pages
and cropped regions under a consistent relevance crite-
rion.Aftereachinspection,onlyquery-relevantobservations,
along with precise source pointers, are recorded in a main-
tained textual ledger. This compact representation preserves
essential evidence while deliberately bounding the accumu-
lation of raw visual history. When exploration terminates
through aconsolidateaction or at the turn limit, the
referenced original page images are reloaded. The reloaded
visual evidence is then consolidated and, when needed, ar-
ranged into a logical sequence for answering. The final an-
swerisgeneratedfromtheseimageswithinthesamerollout,
with indexed entries linking claims to supporting evidence.
ItcanbeseenthatSCOREseparatesexploratorytrialander-
rorfromthefinalevidencechain,reducingexplorationnoise
duringanswergenerationwhilepreservingvisualgrounding.
To further enforce structured evidence utilization, we in-
troduceprocess-levelsupervisionthatexplicitlyshapesinter-
mediate evidence organization rather than only final-answer
correctness.Duringcold-starttraining,ateachermodelgen-
erates trajectories containing relevance decisions, consoli-
dated evidence order, and final answers; only those with
full gold-page coverage and answers judged correct by an
LLM evaluator are retained. In reinforcement learning, an
evidence-awarerewardisproposedtojointlyoptimizegold-
page coverage, compactness of the consolidated chain, and
answercorrectness.Thistwo-stagetrainingpipelineencour-
ages the agent to preserve and organize relevant visual ev-
idence throughout the entire rollout, thereby supporting re-
liable answer generation. To assess the resulting model, we
evaluateSCOREon three established VRAG benchmarks:
ViDoSeek (Wang et al. 2025), SlideVQA (Tanaka et al.
2023), and MMLongBench (Ma et al. 2024).
Our main contributions are summarized as follows:
•WeproposeSCORE,aunifiedagentthatselectsrelevant
visual evidence across pages and regions, then explic-
itly consolidates it by reloading, denoising, and ordering
originalimageswithinthesamerolloutbeforeanswering.
•We introduce process-level supervision: cold-start from
high-coverageteachertrajectoriesandanevidence-aware
RL reward that jointly optimizes gold coverage, consoli-
dation compactness, and answer correctness.
•EvaluatedonthreeVRAGbenchmarks,SCOREachieves
state-of-the-art accuracy across both 3B and 7B back-bones, while ablations confirm the complementary con-
tributionsofselectionandconsolidation,andfurtheranal-
ysesdemonstraterobustretrievalandfavorableefficiency.
Related Work
Visual Retrieval-Augmented Generation
Retrieval-augmentedgeneration(RAG)hasshownstrongef-
fectivenessonknowledge-intensivetasks(Lewisetal.2020;
Gao et al. 2023; Yang et al. 2024), where traditional text-
based approaches generate answers grounded in passages
retrievedfromtextualcorpora.However,withthewideadop-
tion of visually rich documents such as slides, reports, and
scannedPDFs,knowledgeisnolongerconfinedtoplaintext.
This motivates Visual RAG (VRAG), which extends RAG
to retrieve and reason over visual document pages. Early
methods rely on OCR or document parsing to extract text
fromimages,butsuchpipelinesarelossyandfailtopreserve
layout, charts, and figures (Zhang et al. 2025). More re-
cently,OCR-freeretrievalmethodsdirectlyaligntextqueries
with page images: ColPali (Faysse et al. 2024) introduces
late-interactionvisualretrievalviatoken-levelsimilaritybe-
tween query text and image patch embeddings, while EVis-
RAG (Sun et al. 2025) feeds retrieved page images directly
into a VLM for multi-image understanding. However, con-
strained by top-kretrieval, these methods often introduce
irrelevant pages and cannot adaptively refine retrieval based
on prior observations, making it difficult to focus on sparse
query-relevant evidence.
Agentic VRAG with Reinforcement Learning
The agentic RAG paradigm was first introduced by Re-
Act(Yaoetal.2022),whichinterleavesreasoningandaction
sothatthemodelcandecidewhenandhowtoretrieveonthe
fly.Buildingontherecentsuccessofreinforcementlearning
for LLM reasoning (Shao et al. 2024), VRAG-RL (Wang
etal.2026)extendsthisideatovisualdocumentsbydefining
avisualperceptionactionspacewithcroppingandzooming,
andtrainingtheVLMwithmulti-turnRLtoactivelyexplore
evidence across pages. For multi-image evidence manage-
ment,VISOR(Shenetal.2026)maintainsatextualevidence
ledgerduringagentinteraction,recordingquery-relevantvi-
sual observations as text to suppress noise from irrelevant
imagesoverlonghorizons.LAT(Liuetal.2026)insteadtar-
gets evidence attribution, jointly optimizing reasoning and
visual grounding within a single agent via RL, so that an-
swers come with verifiable visual sources. These methods
improve visual exploration, memory, or attribution, but do
notexplicitlyreconstructandgloballyorganizeaselectedset
of original page images before answering. A complemen-
tary line of work decomposes multi-image reasoning across
specialized agents, e.g., ViDoRAG (Wang et al. 2025) as-
signsplanning,retrieval,andansweringtoseparateagentsin
an actor-critic loop. Such pipelines, however, are difficult to
optimizeend-to-endandoftenincuradditionalorchestration
costs.SCOREinstead makes evidence organization explicit
within a unified end-to-end agent loop.

SCORE User  Query
what is
recommended as
a web browser
according to...Update Ledger
Source Pointer &
Relevant ObservationsSearchConsolidate
 Action 
 Evidence
Ledger
(T exual)
ReorderReload  Original  Imagesretrieved image_1
  { The chart shows       
    Microsoft  Edge... ;
    the cropped image...}retrieved image_2 :
  { The image shows       
    nothing related to... }
...
Answer
Generation
 retrieved
image
 cropped
imagexxxx xxxx
 Next T urn Bbox(if evidence > 1)ledger  with  images
"xxxx" : textual 
entries  stored in 
Evidence LedgerFigure 2: Overview ofSCORE. The agent iteratively searches page images or zooms into local regions, updating a maintained
textualevidenceledgerthroughobservationsummarizationandrelevancejudgment.Oncetheevidenceissufficient,thein-loop
consolidateaction reloads the corresponding original images, filters and reorders the retained evidence, and passes the
organized visual context to final answer generation within the same rollout.
Method
Task Formulation
Letqdenote a natural language query andC=
{I1, I2, . . . , I N}a large-scale corpus ofNpage images
extracted from visually rich documents such as slides and
reports. The task is to produce an answeraby iteratively
searchingCand reasoning over the retrieved pages in the
context ofq. Crucially,Cforms a flat, document-agnostic
image pool in which pages from heterogeneous documents
areintermixedwithoutdocument-levelscoping.Relevantev-
idenceansweringqmayresideentirelywithinasinglepageor
bedistributedacrossmultiplepages,whilepagesfromunre-
lateddocumentsserveasdistractors;themodelmustretrieve
and aggregate the relevant visual signals when necessary.
SCOREFramework
As illustrated in Figure 2 and Algorithm 1,SCOREoper-
atesasaunifiedagentloopwithtwopersistentstateobjects:
a maintainedtextual evidence ledgerLand an interaction
historyH. At each reasoning turn, the agent processes the
current page or cropped region and produces a structured
output comprising three fields:⟨observe⟩(a summary of
the visual content),⟨relevant⟩(a binary relevance judg-
ment), and⟨action⟩(the next operation, detailed below).
Relevant observations are added toLalong with pointers to
their corresponding source images, while only the most re-
centWrawvisualobservationsfromHremainintheactive
context. When exploration ends, the original images refer-
enced inLare reloaded and organized for final answering.
Keeping exploration, evidence maintenance, organization,
and answering inside one loop forms a continuous rollout
optimizable with a trajectory-level objective.
Context Construction.Following VISOR (Shen et al.
2026), we control the growth of raw visual context by re-
constructing the input at each turntfrom a fixed prompt,
the evolving evidence ledger, and a bounded window of re-
cent interactions. LetC tdenote the active context suppliedAlgorithm 1:SCOREAgent Loop
Require:Queryq, corpusC, max turnsT, windowW
1:L ← ∅,H ←[ ],(src 1, o1)←(∅,∅)
2:fort= 1toTdo
3:Context←prompt(q),L,currentobservationo t,and
the lastW−1turns ofH
4:Generate(˜o t, ρt, at); att= 1, use
(no image,no,search(q))
5:ifρ t=yesthen
6:Add(src t,˜ot)toL
7:end if
8:Seta t←consolidateift=T
9:ifa t=searchthen
10:(src t+1, ot+1)←top-1 page retrieved fromC
11:else ifa t=bboxthen
12:(src t+1, ot+1)←bbox cropped region
13:else ifa t=consolidatethen
14:E ←original images referenced byL
15:If|E|>1, select, denoise, and orderE
16:Generateansweraction over visual evidenceE
17:returnfinal answera
18:end if
19:Append(o t,˜ot, ρt, at)toH
20:end for
to the agent at turnt; it is formed by concatenating these
components as follows:
Ct=
Pinit;Lt;Ht−W+1:t−1 ;ot
,(1)
whereP initcontains the system prompt and user queryq,
Ltis the ledger of relevance-filtered summaries paired with
source image pointers,H t−W+1:t−1 contains the most re-
centW−1completed turns, ando tis the current page or
cropped region under consideration. We useW= 2to pre-
serve the latest retrieve-then-zoom chain. Raw visual inputs
beyond the sliding window are evicted to bound context us-
age,whilesemanticallydistilledevidencepersistsinL t.The

context includes an intent-injection reminder that restatesq
and points to the collected evidence. Details for the prompt
andintent-injectionreminderareprovidedintheAppendix.
Action space.The agent selects from three structured ac-
tions at each step:(1)⟨search⟩issues a textual query to a
vision-aware retriever and returns the top-ranked page from
Cas the nextretrieved image. The initial search is con-
strained to the original user queryqto prevent premature
query reformulation before sufficient evidence is gathered,
and subsequent searches may generate refined sub-queries
conditioned on the evolving ledgerL t;(2)⟨bbox⟩speci-
fies a bounding box on the current page to obtain a cropped
image, enabling fine-grained inspection of visual regions;
(3)⟨consolidate⟩terminates exploration and triggers
answer generation. It reloads the original full-page images
referenced inL t, reorders and filters them if multiple pages
are retained, and forwards the curated evidence to the final-
answermodule,whichproducestheterminal〈answer〉action
containing the model’s response.
Evidence Ledger: Selection and Consolidation
SCOREorganizessparseevidenceintwostages.Duringex-
ploration, relevant observations, along with pointers to their
source images, are incrementally accumulated in the com-
pact textual ledger; at termination, theconsolidateac-
tionreloadstheoriginalimagesreferencedintheledgerand
globallyreorganizesthemtosupportholisticvisualreasoning
duringanswergeneration.Thisdesignensuresthattheledger
actsasabridgebetweenbounded,sequentialexplorationand
the final, evidence-grounded visual-semantic generation.
Source-linked visual evidence selection.At exploration
turnt, the agent outputs a binary relevance labelρ t∈
{yes,no}along with a textual summary˜o tof the current
visualobservationin<observe>.Theledgerisupdatedby
Lt=Lt−1∥
(srct,˜ot)
, ρt=yes,
Lt−1, ρ t=no,(2)
where∥appends a new entry to the ledger, andsrc tiden-
tifies the original page and, for a cropped observation, its
bbox coordinates. Each retained entry thus links the textual
observation˜o ttoavisualsourcethatcanbereloadedduring
consolidation. Observations markednoare not added toL;
they remain only temporarily in the bounded visual context
and are removed as the window advances.
Global evidence consolidation.Turn-wise relevance
judgments in Eq. (2) may still yield redundant or
suboptimally ordered ledger entries. Upon triggering
⟨consolidate⟩or reaching the maximum turn limitT,
theagentreloadstheoriginalimagesreferencedinthetermi-
nalledgerL τ,whereτdenotesthefinalexplorationstep.For
multipleimages,theagentexaminesthereloadedimagesto-
getherwithrespecttoq,retainstheusefulledgerentries,and
arranges them in a logical order for answering. When only
one image is involved, no further selection or reordering is
needed, and the terminal ledger is used directly. The agent
thenanswersqusingthevisualevidenceassociatedwiththe
resultingledgerentries.Initsfinaloutput,theagentindicateswhich ledger entry supports each claim, thereby linking the
answer to the selected visual evidence rather than the noisy
exploration trajectory.
Training Pipeline
Since per-turn relevance judgment and global consolidation
arenotdirectlysupervisedbyanswercorrectness,weadopta
two-phasetrainingpipeline:acold-startphasethatinstillsthe
structuredoutputformatandbasicledgerbehavior,followed
by areinforcement learningphase whose reward explicitly
targets evidence selection alongside final-answer quality.
Cold-Start via Filtered Trajectory Distillation.We dis-
till completeSCOREtrajectories generated by a stronger
teacher,Qwen3.5-122B-A10B(QwenTeam2026),ontrain-
ing queries from SlideVQA (Tanaka et al. 2023). We retain
onlytrajectorieswithbothLLM-judgedcorrectanswersand
full evidence coverage, requiring every gold reference page
toappearinthefinalanswerledgerratherthanmerelybeing
retrieved.Thisfilterremovessuperficiallycorrecttrajectories
that happen to omit essential supporting evidence. We then
perform SFT with assistant-only label masking to teach the
structured output format and initialize ledger management
and consolidation. Further details on rollout, filtering, and
SFT conversion are provided in theAppendix.
Reinforcement Learning.Building upon the cold-start
checkpoint, we perform multi-turn RL using GRPO (Shao
et al. 2024). SinceSCORErealizes exploration, consolida-
tion, and answering within a single continuous rollout un-
der a unified sliding-window context, a single trajectory-
level optimization suffices to jointly train these capabilities.
This process is driven by a novelevidence-aware trajec-
tory reward, which simultaneously evaluates evidence se-
lection quality and final answer correctness. Formally, let
P(L) ={page(src) : (src,˜o)∈ L}denotethesourcepages
backingaledgerL.WedefineP⋆asthesetofgoldreference
pages andP ans=P(L ans)as the pages selected in the final
answer ledger. We introduce two metrics:
cov=|P⋆∩ Pans|
|P⋆|,cmp=|P⋆∩ Pans|
|Pans|,(3)
wherecov(Recall) rewards the retention of gold pages and
cmp(Precision) rewards a compact ledger by penalizing ir-
relevant retrievals. We definecmp=0 ifP ansis empty. The
trajectory reward combines evidence selection with answer
correctness:
r=cov+λ cmp·Icov=1·cmp+I cov=1·rans−Icov<1·β,(4)
whereλ cmp=0.2andβ=1are hyperparameters, and the in-
dicatorI cov=1equals1whentheledgercoversallgoldpages
and0otherwise(I cov<1isitscomplement).Throughthisin-
dicator, both the compactness term and the answer reward
rans(a binary LLM-as-judge label) are gated on full cover-
age,whileincompletecoverageinsteadincursthepenaltyβ.
This gating prevents the agent from gamingcmpby drop-
ping evidence or receiving answer credit for an incomplete
ledger.Consequently,thisevidence-awaresignalalignswith
SCORE’scoreobjective:compellingtheagenttorecoverall

MethodSlideVQA ViDoSeek MMLongBench
Single-hop Multi-hop Overall Extraction Logic Overall Text Table Chart Figure Layout Overall
Qwen2.5-VL-7B
Vanilla RAG (Faysse et al. 2024) 29.10 17.40 26.10 26.40 41.30 32.88 13.10 14.70 15.90 4.30 7.60 –
ReAct (Yao et al. 2022) 34.80 20.40 31.11 27.50 42.10 33.85 10.10 12.40 10.20 6.20 7.10 –
ViDoRAG†⋆(Wang et al. 2025) 72.15 39.86 63.88 66.05 72.83 69.00 24.40 23.96 21.91 24.14 20.34 25.50
M3RAG†(Du and Li 2026) – – 65.82 – – 69.36 – – – – – –
Search-R1-VL‡(Jin et al. 2025) 48.30 42.30 46.76 40.50 50.30 44.77 19.90 13.40 12.90 11.40 10.20 –
VRAG-RL‡(Wang et al. 2026) 69.30 43.10 62.59 60.60 74.80 66.78 26.10 26.30 24.80 25.90 21.20 –
MMSearch-R1‡⋆(Wu et al. 2025) 52.06 40.21 49.03 55.97 59.56 57.53 16.84 17.97 19.10 18.28 11.86 18.42
EVisRAG‡⋆(Sun et al. 2025) 78.21 42.32 69.09 67.75 72.43 69.79 26.8027.6524.72 22.41 19.49 27.98
R1-Router‡⋆(Peng et al. 2025) 69.66 45.33 63.43 64.19 68.61 66.11 26.46 23.50 22.47 28.28 16.95 26.92
VISOR‡(Shen et al. 2026) 78.82 53.62 72.37 73.4976.66 74.87 27.4923.96 27.53 23.79 22.88 28.45
SCORE(ours)‡82.28 62.26 77.16 73.33 77.87 75.31 32.3025.3531.46 30.00 27.97 31.52
Qwen2.5-VL-3B
Vanilla RAG (Faysse et al. 2024) 19.40 12.20 17.56 10.10 17.30 13.23 2.20 4.10 5.20 4.70 4.30 –
ReAct (Yao et al. 2022) 15.70 10.90 14.47 6.70 14.20 9.96 2.70 3.60 3.40 3.10 5.10 –
ViDoRAG†⋆(Wang et al. 2025) 41.44 19.93 35.94 31.32 38.23 34.33 7.90 6.91 5.06 8.97 7.63 8.50
Search-R1-VL‡(Jin et al. 2025) 26.30 20.10 24.71 20.10 29.80 24.32 8.50 7.80 7.90 9.30 7.60 –
VRAG-RL‡(Wang et al. 2026) 65.30 38.60 58.45 63.1073.8067.76 22.70 16.10 21.90 21.40 19.50 –
EVisRAG‡⋆(Sun et al. 2025) 75.42 47.70 68.35 66.05 72.43 68.82 27.1428.6425.13 24.83 17.80 28.34
R1-Router‡⋆(Peng et al. 2025) 64.93 42.15 59.10 62.64 69.42 65.59 26.80 23.50 21.9125.1721.19 25.74
VISOR‡(Shen et al. 2026) 74.58 50.79 68.49 67.7570.62 69.00 26.80 23.96 26.9723.4522.0327.86
SCORE(ours)‡76.33 53.44 70.47 67.29 71.8369.26 27.4923.96 27.5323.79 21.19 28.51
Table1:MainresultsonSlideVQA,ViDoSeek,andMMLongBench.Wereportaccuracy(%).†denotesmulti-agentarchitectures.
‡denotesfine-tunedmodels.⋆denotesresultsreproducedunderourexperimentalsettingsforafaircomparison;allotherresults
are cited from the original papers. The best result in each column isboldedand the second-best is underlined .
gold pages while filtering distractors, even under sparse and
noisyvisualconditions.DuringRL,retrievedvisualobserva-
tionsandsystem-injectedtokensaremaskedfromthepolicy
loss. The agent decodes from the sliding-window context,
while GRPO aggregates the loss over all agent-generated
tokens under their corresponding rolling contexts using the
shared trajectory-level reward.
Experiments
Experimental Settings
DatasetsandMetric.WeevaluateSCOREonthreeestab-
lishedbenchmarksforVRAG:ViDoSeek(Wangetal.2025),
SlideVQA(Tanaka et al. 2023), andMMLongBench(Ma
etal.2024).Wefollowtheunified-corpusprotocolofVRAG-
RL(Wangetal.2026):alldocumentpagesareflattenedinto
asingleimagepool,andthemodelmustretrievetherelevant
pagesfromthissharedcorpustoanswereachquery,without
any document-level scoping at inference time. Training data
is drawn from the SlideVQA training split and consists of
2,500 trajectories used for cold-start distillation and 1,600
queries used for reinforcement learning. We adopt the same
LLM-judge evaluation as prior work (Wang et al. 2026):
Qwen-max-latest(Qwen Team 2026) compares each
predicted answer with the reference and outputs a binary
correctnessscore,andwereportthemeanasaccuracy.More
details are in theAppendix.
Baselines.We benchmarkSCOREagainst two method
families: (1)vanilla(non-fine-tuned) approaches, including
Vanilla RAG (Faysse et al. 2024), ReAct (Yao et al. 2022),
andmulti-agentpipelinesViDoRAG(Wangetal.2025)andM3RAG(DuandLi2026);and(2)fine-tunedmethodsusing
task-specific supervision likeSCORE: Search-R1-VL (Jin
et al. 2025), VRAG-RL (Wang et al. 2026), MMSearch-
R1(Wuetal.2025),R1-Router(Pengetal.2025),andEVis-
RAG(Sunetal.2025).Toisolateagentdesigncontributions,
all reproduced methods use the ColQwen2.5-v0.1 (Faysse
et al. 2024) retrieval backbone, while adopted results retain
their original settings (compatibility in theAppendix).
Main Results
As shown in Table 1,SCOREachieves the best overall
accuracy across all three benchmarks and both backbone
sizes. Using Qwen2.5-VL-7B as the backbone, it improves
over the strongest prior baseline from 72.37% to 77.16% on
SlideVQA,74.87%to75.31%onViDoSeek,and28.45%to
31.52% on MMLongBench; the same trend holds with the
smaller Qwen2.5-VL-3B. Unlike prior methods that reason
directly over retrieved pages or search trajectories,SCORE
explicitly selects and organizes sparse evidence before an-
swering, leading to consistent gains across benchmarks.
The gains are largest when evidence must be aggre-
gated across pages or localized within complex layouts. On
SlideVQA,SCOREimproves most on multi-hop questions
(62.26% vs. 53.62% from VISOR under the 7B backbone),
while its single-hop gain is more moderate. On MMLong-
Bench, where answers rely on sparse evidence scattered
acrosspagesandconfinedtosmallregions,SCOREachieves
the best 7B results on Text, Chart, Figure, and Layout. This
stems from: (1) selective ledger updates that choose which
pages to retain, and (2) bbox zoom-ins that select which
regions to read—jointly addressing the uneven spatial and

document-level distribution of evidence. Overall, these re-
sults demonstrate the effectiveness of explicitly organizing
sparsevisualevidencebeyondmerelyretrievingitinVRAG.
Ablation Study
Weablatethetwoevidence-organizationmodulesinSCORE:
relevance judgment, which filters observations at collection
time, and consolidation, which globally re-selects and re-
orders the ledger before answering. The variantw/o both
removesboth,degradingSCOREintoaplainaccumulate-all
sliding-windowagent.Wereporteachvariantunderboththe
untrainedVanillasettingandtheFine-tuned(cold-start+RL)
setting,onViDoSeekandSlideVQAwiththeQwen2.5-VL-
7B backbone under the main-result protocol.
VariantSlideVQA ViDoSeek
Vanilla Fine-tuned Vanilla Fine-tuned
SCORE(Full)55.17 77.16 48.25 75.31
w/o relevance judgment 53.50 72.28 47.72 71.80
w/o consolidation 45.24 74.13 36.69 73.91
w/o both 44.06 71.51 35.73 69.79
Table2:Ablationofthetwoorganizationmodules.Accuracy
(%)ontheQwen2.5-VL-7BundertheVanillaandFine-tuned
(cold-start+ RL) settings. Best per column inbold.
Ablating either module hurts performance, and removing
bothis consistently worst, confirming their complementar-
ity. Their relative importanceflipsafter fine-tuning. In the
Vanillasetting, dropping consolidation is far more damag-
ing than dropping relevance judgment (e.g.−9.9vs.−1.7
onSlideVQA):withouttraining,themodeljudgesrelevance
poorly during the intermediate turns, so substantial noise
enterstheledger,makingglobalconsolidationthemorecrit-
icalsafeguardforfilteringitbeforeanswergeneration.After
Fine-tuning,thetrendreverses:droppingrelevancejudgment
becomes the larger drop (e.g.−4.9vs.−3.0on SlideVQA).
Since the model has learned to filter evidence at collection
time, keeping the ledger clean. Without this early filtering,
consolidationmustinsteadprocessraw,unfilteredretrievals.
Ineffect,trainingshiftsthedenoisingresponsibilityfromthe
exit(consolidation)totheentrance(relevancejudgment),yet
both modules remain beneficial throughout. The necessity
of other components ofSCORE, including the maintained
ledger (vs. a sliding window), window size W, intent injec-
tion,andbboxzoom-in,isfurtheranalyzedintheAppendix.
More Analysis
Retrieval and Consolidation Behavior Analysis.We an-
alyzewhereaccuracy comes from using the retrieval and
evidence-organization metrics reported in Table 3. Com-
pleteness measures the percentage of queries whose full re-
trieval trajectory contains every gold page, whereas ledger
coveragemeasuresthepercentagewhosefinalledgerretains
every gold page. Two observations stand out.❶Achieving
highcompletenessincursprohibitivecostswithoutensuring
superior performance. ViDoRAG achieves the highest com-
pleteness (91.3) only by retrieving 10 fixed images, which
introduces many non-gold pages into the visual context. In
4.0 4.5 5.0 5.5 6.0 6.5 7.0
Latency (s)57.560.062.565.067.570.072.575.0Accuracy (%)
EVisRAG VRAG-RLVISORSCORE
SCORE-direct
ViDoRAG
(18.85s, 59%)better
Pareto frontFigure3:AccuracyversusaveragelatencyperqueryonSlide-
VQA.BothSCOREvariantslieontheParetofront,improv-
ing accuracy along the frontier as latency increases.
contrast, VRAG-RL reduces non-gold evidence but misses
relevant pages. This reveals a clearcoverage–noise tension:
modelsstruggletoretrieveallrelevantpageswithoutinclud-
ing noise.❷Explicit evidence organization better resolves
thistension.VISORreducesvisualnoisebyconvertingpages
intoatextualledger,butthisconversionmayintroducetran-
scription errors and, without further selection, still retains
1.29non-gold pages per query. In contrast,SCOREkeeps
only selected pageimages, avoiding conversion loss while
cutting non-gold pages to0.16, about one-fifth of the low-
est image-based baseline. Despite a marginal drop in ledger
coverage, this rigorous filtering yields the highest accuracy
(77.16).Together,theseresultsshowthatselectingandorga-
nizingevidence,ratherthanmerelyretrievingmorepages,is
critical to accurate answering.
Method Comp. Avg. Ret. Ledger Cov. Noise Img. Acc.
EVisRAG (Sun et al. 2025) 80.2 3 (fixed) – 1.97 69.09
VRAG-RL (Wang et al. 2026) 76.8 1.78 – 0.77 62.59
ViDoRAG (Wang et al. 2025)91.310 (fixed) – 8.81 63.88
VISOR (Shen et al. 2026) 84.2 2.3484.21.29 72.37
SCORE81.91.7678.40.16 77.16
Table 3: Retrieval and Consolidation Behavior Analysis on
SlideVQA.Comp.: retrieval completeness (%);Avg. Ret.:
average pages retrieved;Ledger Cov.: gold-page coverage
offinalevidenceledger(%),wheremethodswithoutanevi-
dence ledger are marked “–”;Noise Img.: average non-gold
pages retained for answering;Acc.: answer accuracy (%).
Time Efficiency.We investigate whether the explicit con-
solidationandansweringstepsinSCOREincuraprohibitive
computationalcost.Wehaveappliedtwomechanismstocon-
strain this overhead. First, we bound the raw visual context
to the lastW=2turns, while retaining earlier relevant ob-
servations solely in the textual ledger. Since image tokens
are significantly more expensive than text, this mechanism
prevents visual token consumption from scaling with tra-
jectory length during exploration while preserving essential
evidence. Second, our learned policy converges in an av-
erage of3.65turns, fewer than both ViDoRAG (6.22) and
VRAG-RL(4.36),partiallyoffsettingtheconsolidationcost.
Table4detailstheaverageagentturns,tokenconsumption,
andend-to-endlatencyperquery,showingafavorableaccu-
racy–latency trade-off.SCORE-direct, an efficient variant

Method Avg. Turns Avg. Tokens Latency (s) Acc.
ViDoRAG (Wang et al. 2025) 6.22 23259 18.85 59
VRAG-RL (Wang et al. 2026) 4.36 2932 5.23 61
EVisRAG (Sun et al. 2025)1 2514 4.3761
VISOR (Shen et al. 2026) 3.40 3162 5.68 66
SCORE-direct2.83 2762 4.97 70
SCORE3.65 3468 6.4272
Table 4: Efficiency on SlideVQA (backbone: Qwen2.5-VL-
7B): average agent turns (Avg. Turns), tokens consumed
(Avg. Tokens), and per-query latency (Latency), averaged
over 100 balanced cases (50 single-hop, 50 multi-hop).
ofSCOREthatanswersdirectlyfromthetextualledgerwith-
outvisualconsolidation,achievesanaccuracyof70at4.97s.
This outperforms all baselines in accuracy while remain-
ing faster than all except EVisRAG. Compared to VISOR,
it gains4points while reducing latency by0.71s, suggest-
ing that effective evidence organization contributes signifi-
cantly beyond mere computational overhead. Incorporating
consolidation and answering over reloaded original images
(SCORE) further raises accuracy to72, with average in-
creases of1.45s in latency and706tokens. Notably, against
the multi-agent ViDoRAG,SCOREuses6.7×fewer tokens
while scoring13points higher. Consequently, both variants
reside on the Pareto front in Figure 3.
Effect of Training.To disentangle the contributions of
each training stage, we evaluate three variants ofSCORE
on SlideVQA (Table 5): (i) an untrainedBasethat executes
the framework via prompting alone, (ii) aCold-startcheck-
point obtained through supervised fine-tuning, and (iii) the
fullCold-start+RLmodel.Inadditiontoaccuracy,wereport
retrievalcompleteness(Comp.)andledgercoverage(Ledger
Cov.) to diagnosewhateach stage improves. The dominant
gain originates from cold-start SFT, which lifts overall ac-
curacy from 55.17 to 75.62; RL contributes a further, more
modestincrementto77.16.Weattributethistotheconstruc-
tion of the cold-start data: each filtered trajectory encodes
anexplicitevidence–answercorrespondence,inwhichevery
retained turn records a relevance decision and the consoli-
dation step specifies which pages support the final answer.
Because the filtering criterion admits only trajectories that
yield correct answers while covering all gold pages, SFT
can internalize this alignment directly. Consistent with this
view, ledger coverage exhibits its sharpest increase at the
SFT stage (64.0→76.6), and multi-hop accuracy improves
Variant Comp. Ledger Cov.SlideVQA Acc.
Single Multi Overall
Base (prompting) 72.6 64.0 61.65 36.33 55.17
+ Cold-start (SFT) 80.8 76.6 80.46 61.55 75.62
+ Cold-start + RL81.9 78.4 82.28 62.26 77.16
Table 5: Training-stage decomposition on SlideVQA
(Qwen2.5-VL-7B). Comp.: retrieval completeness; Ledger
Cov.:goldpageskeptintheledger(%).Acc.isoverallaccu-
racy with single-/multi-hop splits. All in %.Base:SCORE
framework by prompting, no training.by 25.2 points (36.3→61.6), confirming that questions
spanning multiple pages benefit from keeping faithful ev-
idence. RL, by contrast, refines evidence selection at the
margin: coverage continues to rise (→78.4) while com-
pletenessremainslargelystatic,suggestingthatRLsharpens
analready-acquiredretrievalpolicyratherthaninducingthe
core behavior from scratch.
Backbone VRAG-RLSCORE∆
Qwen2.5-VL-3B (Bai et al. 2025b) 16.79 28.58+11.79
Qwen2.5-VL-7B (Bai et al. 2025b) 36.98 55.17+18.19
Qwen3-VL-8B (Bai et al. 2025a) 63.97 70.84+6.87
Qwen3.5-27B (Qwen Team 2026) 81.85 82.53+0.68
Qwen3.5-122B-A10B (Qwen Team 2026) 80.72 82.93+2.21
Table 6: Effect of backbone strength on SlideVQA under
aprompting-onlysetting (no training).The trainedSCORE
result is reported in Table 1.
Backbone Analysis.We investigate how backbone capa-
bilityinfluencesthebenefitofSCORE’sexplicitevidenceor-
ganization.Toisolatethiseffect,wecompareagainstVRAG-
RL(Wangetal.2026),arepresentativemethodthataccumu-
lates a retrieval trajectory by iteratively retrieving, zooming
in, and answering from an ever-growing context, across five
backbones spanning two model families (Qwen2.5-VL and
Qwen3/3.5)andscalesfrom3Bto122Bparameters,allunder
a prompting-only setting (no task-specific training). Table 6
reports SlideVQA accuracy and the per-backbone gain∆.
SCOREimproves every backbone, with∆peaking at 7B
(+18.19)andnarrowingonstrongermodels,suggestingthat
they increasingly handle multi-image reasoning and noise
internally.The3Bmodelstillgains11.79pointsbutappears
lessabletoexecuteconsolidationeffectively:underprompt-
ingonly,a3Bvariantthatkeepsthetextualledgerbutdrops
selection and consolidation even scoreshigher(42.26) than
fullSCORE(28.58), as the weaker backbone cannot yet ap-
plythesestepsreliablyandinsteadmis-selectsormis-orders
evidence. The benefit is therefore largest at an intermediate
capacity, where the backbone can follow the explicit struc-
turebuthasnotyetinternalizedit.Beyondaccuracy,SCORE
also turns latent evidence use into readable decisions—
selection, indexed references, and an ordered chain—which
give compact models explicit process-level learning targets.
For Qwen2.5-VL-7B, training then further raises SlideVQA
accuracy to77.16(rowSCORE, SlideVQA Overall in Ta-
ble 1), narrowing the gap to larger backbones.
Conclusion
We presentedSCORE, a unified agentic framework for
VRAG that renders evidence use explicit and controllable.
Atitscore,asingleretrieve–reasonloopfiltersobservations
at collection time, then re-selects and reorders the retained
evidencepriortoanswergeneration,ensuringthateveryrea-
soning step is grounded in a traceable link to its supporting
image. This design controls answer-context noise, and gains
furtherimprovementsfromfilteredcold-startdistillationand
an evidence-selection RL reward that jointly teach evidence
organization across benchmarks and backbones.

References
Arslan, M.; Ghanem, H.; Munawar, S.; and Cruz, C. 2024.
A Survey on RAG with LLMs.Procedia computer science,
246: 3781–3790.
Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.;
Deng,L.;Ding,W.;Gao,C.;Ge,C.;etal.2025a. Qwen3-vl
technical report.arXiv preprint arXiv:2511.21631.
Bai,S.;Chen,K.;Liu,X.;Wang,J.;Ge,W.;Song,S.;Dang,
K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025b. Qwen2.5-VL
Technical Report.arXiv preprint arXiv:2502.13923.
Du, H.; and Li, W. 2026. M3RAG: Orchestrating Multi-
agentReasoningforMulti-hop,Multi-modalUnderstanding.
InInternational Conference on Multimedia Modeling, 364–
378. Springer.
Faysse,M.;Sibille,H.;Wu,T.;Omrani,B.;Viaud,G.;Hude-
lot, C.; and Colombo, P. 2024. Colpali: Efficient docu-
ment retrieval with vision language models.arXiv preprint
arXiv:2407.01449.
Gao, Y.; Xiong, Y.; Gao, X.; Jia, K.; Pan, J.; Bi, Y.; Dai,
Y.; Sun, J.; Wang, H.; Wang, H.; et al. 2023. Retrieval-
augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1): 32.
Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.;
Zamani, H.; and Han, J. 2025. Search-r1: Training llms
to reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal,N.;Küttler,H.;Lewis,M.;Yih,W.-t.;Rocktäschel,T.;
et al. 2020. Retrieval-augmented generation for knowledge-
intensivenlptasks.Advancesinneuralinformationprocess-
ing systems, 33: 9459–9474.
Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao,
C.;Deng,C.;Zhang,C.;Ruan,C.;etal.2024. Deepseek-v3
technical report.arXiv preprint arXiv:2412.19437.
Liu,H.;Li,C.;Wu,Q.;andLee,Y.J.2023.Visualinstruction
tuning.Advances in neural information processing systems,
36: 34892–34916.
Liu, S.; Luo, P.; Zhang, C.; Chen, Y.; Zhang, H.; Liu, Q.;
Kou, X.; Xu, T.; and Chen, E. 2026. Look as You Think:
UnifyingReasoningandVisualEvidenceAttributionforVer-
ifiableDocumentRAGviaReinforcementLearning. InPro-
ceedings of the AAAI Conference on Artificial Intelligence,
volume 40, 32159–32167.
Ma,Y.;Zang,Y.;Chen,L.;Chen,M.;Jiao,Y.;Li,X.;Lu,X.;
Liu, Z.; Ma, Y.; Dong, X.; et al. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding with
visualizations.Advances in Neural Information Processing
Systems, 37: 95963–96010.
Peng, C.; Xu, Z.; Liu, Z.; Li, Y.; Yan, Y.; Wang, S.; Liu, Z.;
Gu,Y.;Yu,M.;Yu,G.;etal.2025. Learningtoroutequeries
across knowledge bases for step-wise retrieval-augmented
reasoning.arXiv preprint arXiv:2505.22095.
Qwen Team. 2026. Qwen3.5 Technical Report. https:
//qwenlm.github.io/blog/qwen3.5/. Hugging Face: https:
//huggingface.co/Qwen.Shao,Z.;Wang,P.;Zhu,Q.;Xu,R.;Song,J.;Bi,X.;Zhang,
H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in open lan-
guage models.arXiv preprint arXiv:2402.03300.
Shen, Y.; Wu, J.; Huang, J.; Yin, D.; Yan, L.; and Cao, M.
2026. VISOR: Agentic Visual Retrieval-Augmented Gener-
ationviaIterativeSearchandOver-horizonReasoning.arXiv
preprint arXiv:2604.09508.
Sun,Y.;Peng,C.;Yan,Y.;Yu,S.;Liu,Z.;Chen,C.;Liu,Z.;
and Sun, M. 2025. VisRAG 2.0: Evidence-Guided Multi-
Image Reasoning in Visual Retrieval-Augmented Genera-
tion.arXiv preprint arXiv:2510.09733.
Tanaka,R.;Nishida,K.;Nishida,K.;Hasegawa,T.;Saito,I.;
andSaito,K.2023. Slidevqa:Adatasetfordocumentvisual
question answering on multiple images. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 37,
13636–13645.
Wang, Q.; Ding, R.; Chen, Z.; Wu, W.; Wang, S.; Xie, P.;
and Zhao, F. 2025. Vidorag: Visual document retrieval-
augmented generation via dynamic iterative reasoning
agents. InProceedingsofthe2025ConferenceonEmpirical
Methods in Natural Language Processing, 9124–9145.
Wang, Q.; Ding, R.; Zeng, Y.; Chen, Z.; Chen, L.; Wang,
S.;Xie,P.;Huang,F.;andZhao,F.2026. Vrag-rl:Empower
vision-perception-basedragforvisuallyrichinformationun-
derstandingviaiterativereasoningwithreinforcementlearn-
ing.Advances in Neural Information Processing Systems,
38: 57133–57160.
Wu,J.;Deng,Z.;Li,W.;Liu,Y.;You,B.;Li,B.;Ma,Z.;and
Liu, Z. 2025. Mmsearch-r1: Incentivizing lmms to search.
arXiv preprint arXiv:2506.20670.
Yang, X.; Sun, K.; Xin, H.; Sun, Y.; Bhalla, N.; Chen, X.;
Choudhary, S.; Gui, R. D.; Jiang, Z. W.; Jiang, Z.; et al.
2024. Crag-comprehensive rag benchmark.Advances in
Neural Information Processing Systems, 37: 10470–10490.
Yao,S.;Zhao,J.;Yu,D.;Du,N.;Shafran,I.;Narasimhan,K.;
and Cao, Y. 2022. React: Synergizing reasoning and acting
in language models.arXiv preprint arXiv:2210.03629.
Zhang, J.; Zhang, Q.; Wang, B.; Ouyang, L.; Wen, Z.; Li,
Y.; Chow, K.-H.; He, C.; and Zhang, W. 2025. Ocr hinders
rag: Evaluating the cascading impact of ocr on retrieval-
augmentedgeneration. InProceedingsoftheIEEE/CVFIn-
ternational Conference on Computer Vision, 17443–17453.

Details of Search Engine
RetrievalisbackedbyColQwen2.5-v0.1(Faysseetal.2024).
All page images are encoded offline into patch-level multi-
vector representations and cached, so that at inference only
the query needs to be embedded. The agent’s textual query
is passed through the same encoder and scored against ev-
ery cached page via the late-interaction MaxSim operator,
yielding a ranked list of candidate pages. Rather than al-
ways returning the single top hit, the environment keeps
a per-trajectory record of pages already shown and returns
the highest-rankedunseenpage as the observation, which
prevents the agent from repeatedly inspecting the same ev-
idence across turns. When every candidate in the top-klist
hasalreadybeensurfacedinpreviousturns,theenvironment
signalsthatnofurtherpagesareavailableandsteerstheagent
toward emitting its final answer.
Necessity of Crop-and-Zoom Tool.
Why cropping is needed.Following VRAG-RL (Wang
et al. 2026) and VISOR (Shen et al. 2026), we bound the
inputresolutionofeverypagethroughafixedmax_pixels
cap instead of feeding pages at native resolution. This keeps
the visual-token cost per image low, but it also means that
onceafullpageisdownsampledtothatbudget,itsdenselocal
content—smallcharts,fineprint,andtightlypackedtables—
becomes illegible. Crop-and-zoom resolves this tension as a
dynamic resolutioncontrol: it re-reads only the sub-region
the agent actually needs at high fidelity, while leaving the
global token budget untouched. Without it, the agent would
be forced to choose between spending its entire budget on
asinglehigh-resolutionpageandreasoningoverunreadable
compressed images; cropping avoids both, trading one extra
turn for exactly the resolution the question demands.
Implementation.The<bbox> [x1, y1, x2, y2]
</bbox>action carries pixel coordinates in the coordi-
nate frame of the image as shown to the VLM, which we
first map through a linear transform back onto the original
high-resolutionpage.Wethenpadtheboxbyafixed28-pixel
margin on all sides, so that content adjacent to the region is
notcutoff,andclampthepaddedboxtothepageboundaryto
avoid out-of-range indices. The resulting region is cropped
and rescaled to a standard resolution, letting the agent re-
solve fine-grained material—dense tables, small charts, and
the like—that is illegible at full-page scale. Malformed or
degeneratecoordinatestriggeranerrormessagethatasksthe
model to reissue the action.
Casestudy.Figure4showsaconcreteinstance.Thequery
“Howmanycommentswereanalyzed?”isansweredbyasin-
gle line of fine print—“Nr. of analyzed comments: 2796”—
tucked beside a dense bubble chart on an otherwise busy
slide. At full-page resolution this line is blurred past the
point of legibility, and the agent misreads it as2798, an an-
swerthatisplausibleandclosebutwrong.Issuinga<bbox>
over that region and re-reading it at high fidelity makes the
digitssharp,andtheagentrecoversthecorrectanswer2796.
The error here is purely perceptual rather than a reasoning
failure:theevidencewasontheretrievedpageallalong,andonly the resolution to read it was missing—exactly the gap
crop-and-zoom is designed to close.
Query:  How many comments were analyzed?
Answer: 2798
Answer: 2796
Figure 4: A crop-and-zoom case study. At full-page resolu-
tion the answer-bearing line is illegible and the agent mis-
reads it as2798; after zooming into the bounding box, the
region is legible and the agent recovers the correct answer
2796. The error is perceptual, not a reasoning failure.
Agent Loop Context
This section expands on the context-construction rule of
Eq.(1)inthemaintextandillustratesitinFigure5.Thedif-
ficultyitaddressesisspecifictomulti-imageagenticVRAG:
a full page image costs a large number of visual tokens,
so naïvely concatenating every retrieved and zoomed image
across a long trajectory exhausts the context budget within
a few turns and, worse, lets early irrelevant images crowd
outthequeryandinducesemanticdrift—theagentgradually
loses track of what it was asked. Following VISOR (Shen
et al. 2026), we resolve this with two coupled mechanisms
operating on a reconstructed per-turn context rather than an
ever-growing transcript.
Persistent ledger plus sliding window.At turntthe con-
text is rebuilt as
Ct=
Pinit;Lt; (rt−W+1 , ot−W+1 ), . . . ,(r t, ot)
,(5)
i.e.thesystempromptandqueryP init,followedimmediately
bythetextualevidenceledgerL t,followedbyonlythemost
recentWraw visual turns. The ledger is pinned right after
thepromptsothateveryrelevancedecisionmadesofarstays
visible in compact textual form, decouplingwhat evidence

has survivedfromhow many raw images are currently in
context. The sliding window then caps the raw-image foot-
printataconstantWturnsregardlessoftrajectorylength:as
shownontheleftofFigure5,turnsolderthanthewindoware
evicted (crossed out), but nothing is truly lost—their query-
relevant content has already been distilled intoL t. We set
W=2because a singleretrieve-then-zoominteraction spans
two turns (asearchthat returns a page, then abboxthat
crops it); keeping thelast two raw turns preserves thatchain
intact while still bounding the image cost.
Intentinjection.Evenwithaboundedwindow,overalong
horizon the sheer volume of intermediate observations can
pull the agent away from the original question. To counter
this, every system-returned observation is augmented with
anintent injectionprompt that restates the queryqand
pointsbacktothecollectedevidence,asdepictedintheTurn
icontextpanel on the right of Figure 5: the user turn pairs
the returned image with an instruction to “refer to the user
queryandthecollectedevidence”wheninterpretingit.This
re-anchors each perception step to the actual objective, so
thatreadinganewpageisalwaysframedbywhattheanswer
requiresratherthanbywhateverthelastfewturnshappened
to surface.
Crucially, the injected prompt isaction-specific: rather
than a single fixed reminder, each type of system return car-
ries a tailored instruction that tells the agent what to do
nextgiventhequeryandtheevidencegatheredsofar.When
asearchreturns a new page, the prompt asks the agent
to judge the page’s relevance toqand, if relevant, distill
its query-pertinent content into the ledger. When abbox
returns a zoomed-in crop, the prompt directs the agent to
read the fine-grained region againstqand update the corre-
spondingledgerentrywithwhatthemagnifiedviewreveals.
Whenconsolidatereturns the reloaded original pages,
the prompt steers the agent to re-examine them jointly—
filtering residual noise and ordering the evidence into a co-
herent chain—rather than treating them as yet another page
to explore. Finally, at theanswerstep the prompt re-states
qalongside the consolidated evidence and asks the agent to
produce the response strictly from it. Tying the injected in-
tenttotheactingstagekeepseveryturnalignednotonlywith
theoriginalquerybutwiththespecificsub-goalofthatturn,
so the agent neither drifts from the question nor blurs the
distinct roles of exploring, consolidating, and answering.
Evaluation Details
Judge Prompt.Following VRAG-RL (Wang et al. 2026)
andVISOR(Shenetal.2026),weuseQwen-max-latest
astheLLMjudgetoscoremodelresponses.Giventheques-
tion, the reference answer, and the model prediction, the
judge returns a binary label (0or1) indicating whether
the prediction is semantically correct. The exact template
is shown in Figure 6.
Reliability of Model-as-Judge.Judging is reliable here
becausethetaskitselfisobjective:everyqueryhasaground-
truth answer that is a short, factual phrase—a year, an in-
stitution, a numeric value—so scoring reduces to check-ing a prediction against a fixed reference rather than mak-
ing an open-ended quality judgment. The reliability of
Qwen-max-latestinthisrolehasalreadybeenvalidated
by VRAG-RL (Wang et al. 2026) and VISOR (Shen et al.
2026). To further corroborate it, we re-evaluate our Slide-
VQAresultswithDeepSeek-V3.2(Liuetal.2024)asan
alternative judge. As shown in Table 7, the two judges pro-
duce highly consistent scores overall, differing only slightly
atthesubtasklevel.Thesmallgapconfirmsthatourreported
results are not sensitive to the choice of judge model.
Judge Single-hop Multi-hop Overall
Qwen-max-latest 82.28 62.26 77.16
DeepSeek-V3.2 81.55 63.67 76.98
Table 7: Comparison of SCORE scores on SlideVQA under
two judge models (Qwen2.5-VL-7B).
Dataset Information
Our evaluation spans three visually rich document bench-
marks, each stressing a distinct evidence regime.
SlideVQA.SlideVQA(Tanakaetal.2023)posesquestions
overpresentationslidesdrawnfromabroadmixofreal-world
decks and topics. Its questions fall into aSingle-hopsubset
(1,648) answerable from one slide and aMulti-hopsubset
(567) that must aggregate evidence across several slides of
the same deck. We evaluate on the complete test split of
2,215questions.
ViDoSeek.ViDoSeek(Wangetal.2025)targetsretrieval-
augmented QA over a large visually rich corpus of roughly
6,000page images mixing text, tables, charts, and figures.
Unlike SlideVQA and MMLongBench, which mix single-
and multi-hop questions, every ViDoSeek question is an-
swerable from a single page: the evidence for each answer
lives on one image, so the challenge is locating that page
ratherthanaggregatingacrosspages.Its1,142testquestions
form two disjoint groups by type:Extraction(645), which
asks the model to locate and read off a specific piece of in-
formation from the retrieved page, andLogic(497), which
additionallyrequiresinferenceorcomputationoverthatcon-
tent to reach the answer.
MMLongBench.MMLongBench(Maetal.2024)empha-
sizes long-context perception over heterogeneous document
content.Keepingonlyquestionswithverifiablereferencean-
swersleaves847evaluationquestions,eachtaggedbycontent
type—Text(291),Table(217),Chart(178),Figure(290),
andLayout(118). A single question may carry more than
onetag,sothesesubsetsoverlapandtheircountsexceed847.
Baseline Implementation Details
Baselinenumberscomefromtwosources.ResultsforVanilla
RAG, ReAct, Search-R1-VL, VRAG-RL, and VISOR are
quoteddirectlyfromtheiroriginalpapers(Faysseetal.2024;
Yao et al. 2022; Jin et al. 2025; Wang et al. 2026; Shen

refer to [user query]
and Collected Evidence  to
[analyse the image/...] 
T i-2Initial
Input
... T i-1 T iEvidence  Ledger
(Texual )              Turn i context 
assistant: <observation>...</observation> <rel....
user:
<imgae>
<intent injection>
 Sliding W indow Context Construct Figure 5: Agent loop context construction. The persistent textual evidence ledger (top) retains the distilled, query-relevant
content of every past turn, while a sliding window (bottom left) keeps only the most recentW=2raw visual turns and evicts
older ones. Each returned observation carries an action-specific intent-injection prompt (right) that restates the query and the
collectedevidenceandtellstheagentwhattodoatthatstep—judgingrelevanceafterasearch,readingthecropafterabbox,
reorganizing evidence afterconsolidate, and answering from the consolidated evidence—so the agent stays anchored to
both the query and the current sub-goal.
 Judge-model Prompt  
System Prompt:
Character Introduction
You are an expert evaluation system for a question answering chatbot.
You are given the following information:
- the query
- a generated answer
- a reference answer
Your task is to evaluate the correctness of the generated answer .
Response Format
Your response should be formatted as following: <judge>T rue or False</judge>
If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to  False.
Please note that the generated answer may contain additional information beyond the
reference answer .
User Prompt:
Query: {Query Description}
Reference Answer: {Reference Answer}
Generated Answer: {Generated Answer}
Figure 6: The prompt template used for LLM-as-Judge evaluation.

et al. 2026), as our evaluation follows exactly the same ex-
perimental protocol, so re-running them would reproduce
the reported figures. M3RAG (Du and Li 2026) is like-
wise quoted from its paper, but under a setup that differs
fromoursinretrievalconfiguration,whichweflagalongside
its result. The remaining four baselines—ViDoRAG, EVis-
RAG,MMSearch-R1,andR1-Router—arereproducedbyus
withinourunifiedevaluationframework,usingColQwen2.5-
v0.1 (Faysse et al. 2024) as the shared retriever so that only
the agent design varies.
VanillaRAG.Asingle-passretrieval-augmentedbaseline:
the original question retrieves relevant pages, which are
handedtothemodelfordirectanswergenerationwithnoiter-
ativereasoningormulti-turninteraction.Weadoptthevisual
variant,inwhichpageimagesareretrievedbyColQwen2.5-
v0.1 (Faysse et al. 2024) and fed straight to the VLM.
ReAct.ReAct (Yao et al. 2022) casts the agent as an in-
terleaved Thought–Action–Observation loop for multi-turn
retrieval-augmented reasoning. Each turn issues a search
queryconditionedonthecurrentreasoningstateandreceives
oneretrievedpageimageastheobservation,iteratinguntila
final answer is produced.
Search-R1-VL.AvisualextensionofSearch-R1(Jinetal.
2025), which brings multi-turn RL-based reasoning into the
RAG loop. The visual variant retargets this framework to
image-based retrieval, trained on the same data with the
same reward and post-processing as VRAG-RL and initial-
ized from a cold-start checkpoint.
VRAG-RL.VRAG-RL(Wangetal.2026)trainsanagen-
tic VLM with GRPO-based RL to iteratively retrieve and
reason over page images. It adds a crop-and-zoom tool for
fine-grainedperceptionandusestrajectory-levelrewardsthat
jointly optimize retrieval and answer quality.
VISOR.VISOR (Shen et al. 2026) is a single-agent
method that maintains a textual evidence ledger over the
interaction, recording query-relevant visual observations as
text to suppress noise from irrelevant images across long
horizons.Asitsharesourexactevaluationprotocol,wequote
its numbers directly from the original paper.
M3RAG.M3RAG (Du and Li 2026) is a multi-agent
framework that splits the retrieval–reasoning pipeline into
specializedagentsformulti-modaldocumentunderstanding.
As its code has not been released, we report its numbers
directly from the original paper.
ViDoRAG.ViDoRAG (Wang et al. 2025) uses an actor–
critic multi-agent architecture with separate planning, re-
trieval, and answering agents for iterative reasoning over vi-
suallyrichdocuments.WepluginColQwen2.5-v0.1(Faysse
et al. 2024) as the single-modal search engine, retrieving
thetop-10pagesperquery,withQwen2.5-VL-7B(Baietal.
2025b) as the backbone VLM; all other components follow
the original pipeline unchanged.
EVisRAG.EVisRAG(Sunetal.2025)performsevidence-
based reasoning over multiple retrieved images, explic-
itly extracting and structuring per-page evidence to sup-portmulti-imageunderstanding.WesubstituteColQwen2.5-
v0.1 (Faysse et al. 2024) for the original VisRAG-Ret re-
triever, retrieving the top-3pages per query, and use the
officially released EVisRAG-7B weights; all other settings
match the original configuration.
MMSearch-R1.MMSearch-R1 (Wu et al. 2025) folds
multimodal search into the reasoning loop via cross-modal
retrievaloverbothvisualandtextualforms.Itshipstwotools,
text search and image-to-image search; in our setting we
adapt text search to retrieve document-page images through
ColQwen2.5-v0.1(Faysseetal.2024),whileimagesearch—
inapplicable to document retrieval—returns a prompt redi-
rectingthemodeltotextsearch.Weusetheofficiallyreleased
MMSearch-R1-7B weights, with all other settings as in the
original.
R1-Router.R1-Router (Peng et al. 2025) uses a dynamic
routingmechanismtrainedwithStep-GRPO:itgeneratesin-
termediatesub-queriesduringreasoninganddispatcheseach
to the most suitable retrieval tool, curbing unnecessary re-
trievalswhileadaptivelyintegratingexternalevidence.Inour
settingeveryretrievaltoolisadaptedtofetchdocument-page
images via ColQwen2.5-v0.1 (Faysse et al. 2024), return-
ing the top-5pages per query, and the interaction budget is
capped at3turns; all other settings follow the official con-
figuration.
Compute.All experiments run on8×NVIDIA A800
80GB GPUs.
Training Hyperparameters
Name Value
Finetuning type Full
Freeze vision tower True
Freeze multi-modal projector True
Freeze language model False
Cutoff length 16384
Epochs 3
Batch size 16
Gradient accumulation steps 2
Learning rate 1.0e-5
LR scheduler type cosine
Warmup ratio 0.1
Table 8: Key hyperparameters for cold-start SFT (shared by
the 7B and 3B backbones).
Table 8 and 9 list the full hyperparameter settings for the
cold-start SFT stage and the RL stage, respectively. We de-
liberatelykeepasingleconfigurationacrossbothbackbones:
the Qwen2.5-VL-7B and 3B models are trained under iden-
tical hyperparameters, so that any performance difference
betweenthem reflectsbackbonecapacity ratherthantuning.
All runs use one node of8×NVIDIA A800 GPUs. The

Name Value
Number of agent groups 5
Warmup steps ratio 0.285
Train batch size 8
Mini batch size per GPU 1
Micro batch size per GPU 1
Learning rate (Actor) 1.0e-6
KL loss coefficient 0.01
Tensor model parallel size 2
Max prompt length 8192
Max response length 2048
Max turns 10
Total steps 100
GPU memory utilization 0.4
Table 9: Key hyperparameters for RL (shared by the 7B and
3B backbones).
settings are otherwise not tuned per benchmark; when port-
ingSCOREtosubstantiallylargerorsmallerVLMs,scaling
learningrateandcontextlengthaccordinglyislikelytohelp.
Details of Cold-Start
We build the cold-start corpus with an automatic teacher-
rollout pipeline on SlideVQA training queries. Because the
teacher follows the same SCORE action protocol and the
same ledger-plus-sliding-window context used at inference,
wedonotrestatethosemechanicshere(seetheMethodsec-
tion and the Agent Loop Context section above) and instead
focusonwhatisspecifictogeneratingandfilteringthedata.
Teacher Rollout.For each query we prompt a stronger
teacher, Qwen3.5-122B-A10B, to act as a SCORE agent
and interact with the retrieval environment under the fixed
action schemaobserve,relevant,{search,bbox,
consolidate} defined in the main text: at each step it
reads the current image, marks whether that image carries
query-relevantevidence,andemitsexactlyonenextaction—
a refinedsearchwhen evidence is still insufficient, a nor-
malizedbboxwhen a relevant page hides answer-critical
localdetail(tablecells,chartvalues,axes,smalltext,names,
dates, or numbers), orconsolidateonce enough evi-
dence is in hand. The environment logs both the teacher
messages and its returned visual observations: asearch
appends the retrieved page image, and abboxappends the
cropped region as the next observation. To bound visual to-
ken cost, every image handed to the teacher is resized to
a fixed pixel budget, constrained between256×28×28
and512×28×28. Each trajectory is capped at a maxi-
mum number of interaction steps; if the teacher does not
consolidate within that budget, the environment forces con-
solidationovertheevidencecollectedsofarandproceedsto
answer generation. The resulting trajectory thus records the
complete trace: per-turn structured reasoning, retrieval ac-tions, optional region zooming, the consolidation index list,
and the final answer.
Trajectory Filtering.Teacher rollouts can still contain
noisy or shortcut solutions—most dangerously, answers
judgedcorrectwithoutactuallyvisitingalltheevidencethey
should depend on. We therefore keep a trajectory only if
it passes two conjunctive checks.(i) Answer correctness.
An LLM judge receives the question, the reference answer,
and the teacher’s final answer and returns a binary label;
it is instructed to accept semantic, numerical, and abbrevi-
ation equivalences (e.g., treating “2Bn” and “2 billion” as
thesame).(ii)Fullevidencecoverage.FromeachSlideVQA
sample we build the set of gold reference pagesP gold(from
thesourcefilenameandtheannotatedreferenceindices)and
the set of pages retained in the trajectory’s evidence ledger
Pledger—i.e.,thesourcepagesoftheobservationstheteacher
markedrelevant=yesand kept—and require
JudgeCorrect(a, a⋆) = 1∧ P gold⊆ P ledger.
Samples lacking reference-page metadata are exempt from
the coverage test rather than discarded. This filter is deliber-
atelyconservative:thejudgeremoveswrong-answertrajecto-
ries,whilethecoverageconstraintremovesfalsepositives—
trajectories that match the answer withoutretainingevery
annotated page as evidence—so the student is less likely to
imitate spurious retrieval or incomplete evidence selection.
Note that coverage is checked against the kept ledger rather
than every page the teacher merely glanced at: a gold page
thatwasretrievedbutdiscardedasrelevant=nodoesnot
count, which is exactly what makes the criterion supervise
selectionand not just retrieval.
SFTConversionandSupervision.Eachsurvivingtrajec-
tory is converted into a multimodal chat-style SFT instance:
textiskeptverbatim,whileeveryenvironment-providedim-
age is replaced by an<image>placeholder aligned to a
separate image list. We drop malformed instances—empty
assistantturns,trajectorieswhosefinalmessageisnottheas-
sistant’s, and those exceeding a fixed image budget—while
preservingthefullagenticformat(initialinstruction,per-turn
teacher outputs, returned or cropped images, consolidation
output, and final answer). For training simplicity, cold-start
supervisesthecompletetrajectory:unlikeinferenceandRL,
where the context is reconstructed per turn with only the
ledger plus the lastWraw visual turns, the SFT instance
keeps every returned image in place, with neither a sliding
window nor a dynamically rebuilt ledger injected into the
context. What the student is explicitly supervised to repro-
duce is therefore three behaviors along the raw trace: (i) the
multi-turnsearch/observeretrieval-and-reading loop;
(ii) the per-imagerelevant=yes/nodecision that fil-
tersevidenceatcollectiontime;and(iii)atconsolidate,
thererankovertheevidenceaccumulateduptothatpointto-
getherwiththefinalanswergeneratedfromit.Learningthese
on the full rollout lets the student internalize the ledger and
consolidation behavior before the sliding-window context is
imposed at deployment. Training uses assistant-only super-
vision: loss is computed solely on the teacher’s structured

generations (observe,relevant, action decisions, con-
solidation indices, and answer), with user instructions and
environmentobservationsservingascontextonly.Thisstage
teaches the student the SCORE protocol, ledger behavior,
and consolidation format, providing a reliable initialization
before RL further sharpens query-aware evidence selection.
On an Evidence Re-Fetch Module
Motivation.Retrieval order in a multi-hop query is not
deterministic,andthisraisesaconcernaboutourcollection-
time relevance judgment. Consider a question such as“for
thecompanyrankedthirdbymetricAin2010,whatisitsmet-
ricBin2014?”Theretrievermaysurfacethe2014metric-B
pagebeforethe 2010 ranking is known, so at the moment
that page is inspected the agent cannot yet verify whether it
concerns the target company. A strict relevance filter might
then discard it as irrelevant and, because that discard is per-
manent, lose a gold page it will later need. This suggests
anevidence re-fetchmodule: after consolidation, allow the
agenttoreconsiderandpullbackpagesithadearlierdropped,
once the query constraints have become clear.
Finding.Weimplementedsuchamoduleandfoundthatit
does not help and in fact slightly hurts (Table 10). Two ob-
servations explain why. First, the relevance judgment rarely
discards gold pages to begin with: SCORE already keeps
a ledger coverage of78.4(main-text analysis), so the feared
“prematurediscard”isuncommon—whenuncertainwhether
apagequalifies,thetrainedpolicydefaultstokeepingitrather
than dropping it, and the ledger errs toward over-retention,
exactly the safe direction for recall. Consistent with this,
adding re-fetch barely moves coverage at all (78.4→78.5):
thereisalmostnothinglefttorecover.Second,andmoredeci-
sively,thefewpagesthataredroppedcannotbebroughtback
usefully. Inspection shows these are not order artifacts but
genuine perception failures—the model misread the page’s
content, so it would misuse the same page even if handed
it back. Worse, letting the agent reopen previously rejected
pages reintroduces the very noise the relevance filter was
meant to remove, nudging a few borderline cases from cor-
recttowrong,sooverallaccuracyedgesdownratherthanup
(77.16→76.79).
Variant Ledger Cov. Acc.
SCORE 78.477.16
+ evidence re-fetch 78.5 76.79
Table 10: Effect of adding a post-consolidation evidence re-
fetch module on SlideVQA (Qwen2.5-VL-7B). Ledger Cov.
isthefractionofgoldpageskeptintheledger.Re-fetchleaves
coverage essentially unchanged—there is little to recover—
while slightly lowering accuracy by re-admitting previously
rejected pages.
Key findings.We therefore drop the re-fetch module. Un-
der our current benchmarks the relevance judgment already
keeps coverage high, the pages it does drop are lost to mis-
perception rather than to retrieval order—so re-fetch recov-ersalmostnothing(+0.1coverage)—andreopeningrejected
pagesonlyletsnoisebackin,slightlyloweringaccuracy.We
note, however, that these benchmarks may be relatively be-
nignforthisquestion:longerhorizons,morehops,orharder
constraint-orderingcouldmakeprematurediscardsmorefre-
quent and a well-designed re-fetch more valuable, and we
leave that exploration to future work.
Main-Text Ablation Details
Training the Ablation Variants
The main-text ablation (Table 2) reports each variant un-
der both an untrainedVanillasetting and aFine-tuned
(cold-start+RL) setting. To avoid confounding architecture
with training, everyFine-tunedvariant isretrained from
scratchunder the exact same pipeline as full SCORE—the
same teacher, cold-start filtering, GRPO configuration, and
hyperparameters—rather than obtained by disabling a mod-
ule at inference time on the full model. The two structural
designs, the sliding-window context and the persistent ev-
idence ledger, are retained inallvariants; what changes is
only which of the two agenticactions—relevance judgment
and consolidation—the policy is allowed to emit.
What each variant does.Concretely, all variants reload
the original page images referenced by the ledger before
answering; they differ only in whether relevance judgment
and evidence consolidation are performed:
•w/o relevance judgment.The per-turn<relevant>
yes/no </relevant>decision is removed, so each
turn produces only an observation and the next action;
every retrieved page is written into the ledger unfiltered.
Before answering, the original images referenced by the
ledger are reloaded and consolidation still re-selects and
reorders them.
•w/o consolidation.The global re-selection, denoising,
andreorderingstepisremoved.Whentheagenthasgath-
eredsufficientevidence,itreloadstheoriginalimagesref-
erenced by the accumulated ledger and answers directly
from all of them in their original collection order.
•w/o both.Neither relevance judgment nor consolidation
isavailable.Everyretrievedpageiswrittenintotheledger
unfiltered;oncetheagentjudgesthegatheredinformation
sufficient, it reloads all original page images referenced
by the ledger and answers from them in their original
collection order, without collection-time filtering or pre-
answer organization.
Reward.AllFine-tunedvariantskeepthefullledger-based
reward of Eq. 4, including the coverage and compactness
terms. We deliberately donotweaken the reward for the ab-
lated variants: the evidence-selection signal is applied iden-
tically in every case. For variants without consolidation the
reward is computed on the final accumulated ledger (which
coincides with the consolidated ledger when consolidation
is absent), so every variant is optimized toward the same
evidence-coverage objective and any accuracy gap reflects
the missing action rather than a different training target.

Additional Ablations
Both ablations in this section are conducted on SlideVQA,
the source of our training data, and we report accuracy (%)
on its test set.
Persistentevidenceledger.Thetextualevidenceledgeris
a core mechanism of SCORE and cannot simply be deleted,
sinceconsolidation,thefinalanswer,andtherelevancejudg-
ment all build on it. What we test here is instead its role
duringiterativeretrieval.Thisrequiresremovingthesliding
windowatthesametime:theledgerandtheraw-imagewin-
dow are SCORE’s two carriers of cross-turn memory, so if
we dropped only the textual ledger, the recent images still
visibleinthewindowwouldsilentlystandinforitandmask
its effect. Removing both leaves the agent with no explicit
memoryofwhatithasalreadyfoundacrossturns—eachstep
seesonlythecurrentobservation—whichisexactlythecon-
dition that isolates the ledger’s contribution to long-horizon
retrieval. The agent can still mark each observed page as
relevant or not, and consolidation still runs before answer-
ing.However,withoutarunningtextualmemorytheretrieval
process starts todrift: lacking a compact anchor of the ev-
idence gathered so far, the agent brings in more and more
noiseasthetrajectorygrows,anditssuccessivequerieswan-
derawayfromthetarget.Eventhoughrelevancefilteringand
finalconsolidationarestillpresent,thisdrifthappensduring
collection,soincasesthatneedseveralretrievalattemptsthe
agent may never surface the key image at all. As shown in
Table 11, removing the iterative textual ledger lowers final
accuracy, confirming that maintaining evidence as running
text—not only consolidating it at the end—is what keeps
long-horizon retrieval on track.
Variant Acc.
SCORE 77.16
w/o iterative ledger 73.09
Table 11: Effect of removing the iterative textual evidence
ledger (together with the sliding window).
Sliding-window context.As an additional study, we keep
thetextualevidenceledgerandvaryonlytheraw-imagewin-
dow size, comparingW=1,2,3against aw/osetting that
keeps no raw image at all. A smaller window is cheaper
but may drop the page needed to interpret a recent crop; a
larger window keeps more images but also brings more vi-
sualtokensandirrelevanthistoryintotheprompt.Asshown
in Table 12,W=2gives the best trade-off: it preserves the
commonretrieve–zoompairwhileleavingolderevidenceto
the ledger.
Prompt Template
Figure 7 summarizes the prompt templates used by SCORE
throughouttheagentloop.Thetemplatecontainsthesystem
instruction,thetextualevidencespace,action-specificobser-
vation prompts, the reranking prompt for consolidation, andMetricW=1W=2W=3w/o
Acc. 76.66 77.16 76.16 69.21
Table 12: Effect of the sliding-window size (textual ledger
kept in all settings).
thefinal-answerprompt.Bluefieldsdenoteruntimevariables
filledbytheenvironment,suchastheuserquestion,returned
visual tokens, image file names, and evidence entries.
Case Study
Good case.Figure 8 shows a representative multi-hop ex-
ample where SCORE’s relevance filtering and consolida-
tion work together. The question asks for Nestlé’s Trading
Operating Profit in the year with the third-largest Organic
Growth over a ten-year period. During collection, SCORE
doesnottreateveryretrievedNestléslideasuseful:itrejects
an operational-efficiency slide that does not contain either
the growth ranking or the profit value, while keeping the
financial-performance slide and the ten-year growth chart.
After the growth chart identifies 2011 as the target year,
consolidation reorders the retained evidence into a readable
chain: first the ranking page that establishes the year, then
the 2011 financial page that gives the profit. The final evi-
dence table and answer therefore expose the reasoning path
directly, rather than leaving the reader to inspect the full
retrieval history.
Bad case.Figure 9 shows a failure mode where retrieval
succeeds but visual structure understanding fails. The agent
correctly retrieves the divestiture slide and identifies that
Finduswasdivestedin2000.Italsoretrievestherightacqui-
sitionslideforthesameyear.However,theacquisitionchart
places multiple brand logos around the 2000 bar, including
PowerBar and Purina, and the model incorrectly associates
the year with Purina instead of the target brand PowerBar.
Thus the error is not caused by missing evidence or search
failure: the answer-bearing image is already present in the
trajectory, but the model fails to parse the chart layout and
bind the correct logo to the year. This case suggests that
SCORE’s evidence selection and consolidation can expose
the right pages, yet fine-grained structural perception inside
a dense visual chart remains a limiting factor.

 Prompt T emplate 
System Prompt:
You are a visual reasoning agent. You will search for images to answer the user's question.
For every response, output exactly three parts in order:
Part 1 — <observe>your analysis</observe>
Carefully read ALL data in the image. Actively derive intermediate conclusions — if the image contains multi-year data, rank or compare values yourself to identify the needed year or value, and state the
conclusion explicitly .
Part 2 — <relevant>yes</relevant> or <relevant>no</relevant>
The ﬁnal answer may require combining information from MUL TIPLE images. Mark an image relevant if it contributes to ANY step of the reasoning chain — even if it only answers part of the question. Only
mark no if the image has absolutely no connection to the question.
- yes: the image helps resolve ANY step, including intermediate steps (e.g. identifying a year , a name, a value needed later)
- no: the image has absolutely no useful information for any step
Part 3 — exactly one action:
- <search>query</search> — retrieve a new image. Use the original question as the search query; only reﬁne it when you have a speciﬁc intermediate target (e.g. a year or name derived from prior evidence).
- <bbox>[x1,y1,x2,y2]</bbox> — zoom into a speciﬁc region when necessary (0–1000 normalized, only on full non-cropped images).
- <consolidate/> — triggers reranking and ﬁnal synthesis. ONL Y use this when evidence is truly sufﬁcient to answer with conﬁdence.
Rules:
- NEVER skip any of the three parts.
- Output exactly one action per response.
- Always prefer <search> over giving up. If the current image is unhelpful, search again with a reﬁned or dif ferent query .
- Only trigger <consolidate/> when you have collected enough evidence to answer conﬁdently , or when instructed by the system.
- NEVER answer directly — always use <consolidate/> to ﬁnalize your answer .
Good examples:
<observe>This slide shows Nestlé's 10-year Organic Growth history with values for each year . Ranking them: 2007 had 7.5% (1st), 2010 had 7.2% (2nd), 2005 had 6.8% (3rd). So the third largest Organic
Growth was 6.8% in 2005. I now need the Trading Operating Proﬁt for 2005.</observe><relevant>yes</relevant><search>Nestlé Trading Operating Proﬁt 2005</search>
<observe>This is a product portfolio slide. No ﬁnancial performance data is present.</observe><relevant>no</relevant><search>Nestlé Trading Operating Proﬁt 2005</search>
<observe>I now have Trading Operating Proﬁt for 2005 (CHF 9.8B) from the collected evidence, and conﬁrmed 2005 is the year of the third largest Organic Growth. Enough to answer .</observe>
<relevant>yes</relevant><consolidate/>
Question:  {Query Description}
Image Observation Prompt with Intent Injection :
{vision_tokens} Analyze this image for the question: [ {Query Description} ]. Output <observe>, <relevant>, and your next action.
Crop Observation Correction Prompt :
{vision_tokens}  This is the cropped image. Analyze it for [{question}]. Output <observe>, <relevant>, then <search> or <consolidate/> (no more bbox).Evidence Space (T exual) :
[1] File:  {image_basename_1}
  {observation_1}
[2] File: {image_basename_2}
  {observation_2}
Final Answer Prompt :
[1] File:  {ﬁlename_1}
Evidence: {thought_1}
{image_1}
[2] File: {ﬁlename_2}
Evidence: {thought_2}
{image_2}
Based on the reranked evidence above, answer the question: [ {question} ]
Carefully read each image directly — images are the primary source of truth.
Output format: <observe>your reasoning</observe><answer>ﬁnal answer</answer>Rerank Prompt :
[1] File: {ﬁlename_1}
Evidence: {thought_1}
{image_1}
[2] File: {ﬁlename_2}
Evidence: {thought_2}
{image_2}
Question: [ {question} ]
Review the evidence entries above. Each entry has an index, ﬁlename, evidence text, and image.
Rerank the entries by relevance and logical reasoning order for answering the question.
Output format: <observe>your reasoning</observe> followed by a JSON list of ALL 1-based indices in order of relevance, e.g. [3, 1, 2].
Include ALL entries. Output nothing else.
No More New Images Prompt :
No more new images. Use <consolidate/> to ﬁnalize.
Invalid Action Prompt :
Your previous action is invalid. Output <observe>, <relevant>, and one of <search>, <bbox>, <consolidate/> or <answer>. Try again.Crop Failed Prompt:
Crop failed. Output <observe>, <relevant>, and one actionFigure 7: Prompt template of SCORE. Blue fields are runtime variables inserted by the environment.

 Good Case 
User Query:  How much is the Trading Operating Proﬁt in the year Nestlé achieved the third largest Organic Growth in 10 years?
Reference Answer:  12.5 bn
Step 1: 
<observe>The user is asking for the Trading Operating Proﬁt of Nestlé in a speciﬁc  year. The year is deﬁned as the one where Nestlé achieved the third largest Organic Growth in
its 10-year history . I have no evidence yet to identify this speciﬁc year or the corresponding proﬁt ﬁgure. I need to ﬁrst ﬁnd the list of Nestlé's Organic Growth percenta ges over the
past 10 years to rank them and identify the third largest value.</observe>
<relevant>no</relevant>
<search>How much is the Trading Operating Proﬁt in the year Nestlé achieved the third largest Organic Growth in 10 years?</search>
Step 2: 
<observe>The image displays ﬁnancial data for the year 2011, speciﬁcally under the title \"201 1: Strong Full Year
Performance\". It lists \"Organic Growth\" as +7.5%. This is the second slide in a sequence (as indicated by the date
February 21st, 2012, which is the next day after the presentation date on this slide). The question asks for the year
with the third largest Organic Growth in a 10-year period. Since this slide shows 2011 had +7.5% Organic Growth, it
is not the year with the third largest growth (which would be larger than 7.5%). Therefore, this image does not help
identify the target year. However , it provides context for the type of data need ed (Organic Growth and Trading
Operating Proﬁt). I need to ﬁnd the full 10-year history to rank the growths.</observe>
<relevant>yes</relevant>
<search>Nestlé 10-year Organic Growth history ranking</search>
Step 3: 
<observe>The image displays a timeline of Nestlé's Operational Efﬁciency initiatives from 1997 to 2011. It shows key
years like 2008 and 2011, but does not contain  Nestlé 10-year Organic Growth history ranking.  I need to ﬁrst ﬁnd the
list of Nestlé's Organic Growth percentag es over the past 10 years to rank them and identify the third largest value.
</observe>
<relevant>no</relevant>
<search>Nestlé 10-year Organic Growth history ranking</search>
Consolidation: 
<observe>The question asks for the Trading Operating Proﬁt in the year when Nestlé achieved the third largest Organic Growth over the 10-year period. The evidence should
be ordered according to the reasoning chain needed to answer this question.
Entry [2] should come ﬁrst because it contains the chart \"10 years of the Nestlé Model\" with Organic Growth shown by the blue bars. Ranking the Organic Growth values, the
largest is 9.7% in 2001, the second largest is 8.3% in 2008, and the third largest is 7.5% in 201 1. Therefore, Entry [2] identiﬁes the target year as 201 1.
Entry [1] should come next because it is the 2011 performance slide and directly gives the Trading Operating Proﬁt for 2011 as 12.5 bn. This completes the second step of the
reasoning chain.
Therefore, the useful evidence should be ordered as [2, 1].</observe>
[2, 1]
Step 4: 
<observe>The image displays a chart titled "10 years of the Nestlé Model" showing Organic Grow th as the blue bars and EBIT  margin as
the line. The question asks for Organic Growth, so I should rank the blue bar values, not the EBIT  margin line values. The Organic
Growth values shown are:\n- 2001: 9.7%\n- 2002: 4.9%\n- 2003: 5.1%\n- 2004: 4.5%\n- 2005: 6.2%\n- 2006: 6.2%\n- 2007: 7.4%\n-
2008: 8.3%\n- 2009: 4.1%\n- 2010: 6.2%\n- 2011: 7.5%\n\nRanking the Organic Growth values:\n1st: 9.7% (2001)\n2nd: 8.3%
(2008)\n3rd: 7.5% (2011)\n\nSo, the year with the third largest Organic Growth is 2011. I already collected evidence from the 2011
performance slide showing Trading Operating Proﬁt is 12.5 bn, so the evidence is sufﬁcient to answer .</observe>
<relevant>yes</relevant>
<consolidate/>
FInal Answer: 
<observe>The reranked evidence supports a two-step reasoning process. First, Entry [1] shows the
"10 years of the Nestlé Model" chart, where Organic Growth is represented by the blue bars. Ranking
the Organic Growth values, the largest is 9.7% in 2001, the second largest is 8.3%  in 2008, and the
third largest is 7.5% in 2011. Therefore, the target year is 2011. Second, Entry [2] is the 2011 full year
performance slide, which lists Trading Operating Proﬁt as 12.5 bn. Thus, the Trading Operating Proﬁt
in the year with the third largest Organic Growth is 12.5 bn.</observe>
<answer>12.5 bn</answer>Figure 8: A good case showing relevance filtering and evidence consolidation.

 Bad Case 
User Query:  What is the name of the brand which Nestlé acquired in the year Findus was divested?
Reference Answer: PowerBar
Step 1: 
<observe>The user is asking for the name of a Nestlé brand acquired in the same year that Findus was divested. I need to ﬁrst identify the year Findus was divested and then ﬁnd
the corresponding acquisition by Nestlé in that same year . No evidence has been collected yet.</observe>
<relevant>no</relevant>
<search>What is the name of the brand which Nestlé acquired in the year Findus was divested?</search>
Step 2: 
<observe>The image displays a timeline of \"Divestitures 2000 to 2011\". I can see a bar labeled \"-Findus EUR\"
starting in the year 2000. This indicates that Findus was divested in 2000. The question asks for the brand acquired
by Nestlé in the *same* year (2000) that Findus was divested. Looking at the timeline, there are no acquisitions
shown in 2000. However , there is a bar labeled \"-Mitsag Water GER\" in 2000, which suggests a divestiture of
\"Mitsag Water\". The question speciﬁcally asks for the *acquisition*, not a divestiture. Therefore, I need to search for
what Nestlé acquired in 2000.</observe>
<relevant>yes</relevant>
<search>Nestlé acquisitions 2000</search>
Step 3: 
<observe>The image displays a timeline of \"Acquisitions 2000 to 2011\". I can see a bar labeled \"PURINA\" starting
in the year 2000. This indicates that Nestlé acquired PURINA  in 2000. The previous evidence conﬁrmed that Findus
was divested in 2000. Therefore, the brand acquired by Nestlé in the same year (2000) that Findus was divested is
PURINA.</observe>
<relevant>yes</relevant>
<consolidate/>
Consolidation: 
<observe>The question asks for the brand acquired by Nestlé in the same year that Findus was divested. From entry [1], we learn that Findus was divested in 2000. Entry
[2] then shows that Nestlé acquired PURINA  in 2000. Therefore, the logical order is to ﬁrst identify the divestiture year (Entry [1]) and then ﬁnd the corresponding acquisition
(Entry [2]). Both are necessary and directly relevant.</observe>
[1, 2]
FInal Answer: 
<observe>The question requires two steps: ﬁrst, identifying the year in which Findus was divested,
and second, ﬁnding the brand acquired by Nestlé in that same year. From entry [1], the timeline
shows \"-Findus EUR\" starting in 2000, indicating Findus was divested in 2000. Entry [2] then
conﬁrms that Nestlé acquired PURINA  in 2000. Thus, the brand acquired in the year Findus was
divested is PURINA.</observe>
<answer>PURINA</answer>
Figure 9: A bad case where the correct image is retrieved but chart-structure understanding leads to the wrong answer.