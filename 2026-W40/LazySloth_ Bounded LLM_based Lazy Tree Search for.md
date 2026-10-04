# LazySloth: Bounded LLM-based Lazy Tree Search for Fast Long Video Comprehension

**Authors**: Arka Mukherjee, Kaleen Shrestha, Larissa Zhu, Maja Matarić

**Published**: 2026-09-29 12:53:57

**PDF URL**: [https://arxiv.org/pdf/2609.37426v1](https://arxiv.org/pdf/2609.37426v1)

## Abstract
Modern vision-language models (VLMs) have shown promising results in long-video understanding due to the rich semantic information they can capture. However, most methods focus on coarse captioning of extracted image frames that are computationally inefficient and require models with large context windows. While past work has explored efficient methods through multimodal retrieval-augmented generation (RAG), they rely on lossy embeddings that lose temporal context and fine-grained detail. Few works to date have investigated how VLM-based query-relevant information retrieval can be optimized. We introduce LazySloth, an efficient tree-based search method that speeds up video comprehension and retrieval tasks 2.9-8.3x (compared to existing agentic methods) through bounded captioning of portions of the video considered irrelevant by a VLM of the video. Compared to contemporary specialized video-understanding VLMs and RAG-based methods, LazySloth achieved similar or better final task accuracy across two recent open-source base VLMs--Gemma 4 31B and Qwen3.6 27B--across four benchmarks. LazySloth reduced the gap between the base open-source model and a closed-source model, GPT-4o. Ablations showed that replacing VLM scene understanding with CLIP-based retrieval cost 8.8-19.9% in accuracy, while lazy tree construction matches eager construction at a fraction of the captioning cost. With LazySloth, we demonstrate the possibility of faster long-video comprehension without substantial loss in performance.

## Full Text


<!-- PDF content starts -->

LazySloth: Bounded LLM-based Lazy Tree Search
for Fast Long Video Comprehension
Arka Mukherjee1, Kaleen Shrestha2, Larissa Zhu2, Maja Matarić2,
1School of Computer Engineering, Kalinga Institute of Industrial Technology (KIIT) Bhubaneswar
2Computer Science Department, Viterbi School of Engineering, University of Southern California
Abstract
Modernvision-languagemodels(VLMs)haveshownpromis-
ing results in long-video understanding due to the rich se-
manticinformationtheycancapture.However,mostmethods
focusoncoarsecaptioningofextractedimageframesthatare
computationallyinefficientandrequiremodelswithlargecon-
textwindows.Whilepastworkhasexploredefficientmethods
through multimodal retrieval-augmented generation (RAG),
they rely on lossy embeddings that lose temporal context
and fine-grained detail. Few works to date have investigated
how VLM-based query-relevant information retrieval can be
optimized. We introduce LazySloth, an efficient tree-based
search method that speeds up video comprehension and re-
trieval tasks 2.9-8.3x (compared to existing agentic methods)
through bounded captioning of portions of the video consid-
ered irrelevant by a VLM of the video. Compared to con-
temporaryspecializedvideo-understandingVLMsandRAG-
basedmethods,LazySlothachievedsimilarorbetterfinaltask
accuracyacrosstworecentopen-sourcebaseVLMs–Gemma
431BandQwen3.627B–acrossfourbenchmarks.LazySloth
reduced the gap between the base open-source model and a
closed-source model, GPT-4o. Ablations showed that replac-
ingVLMsceneunderstandingwithCLIP-basedretrievalcost
8.8–19.9% in accuracy, while lazy tree construction matches
eager construction at a fraction of the captioning cost. With
LazySloth,wedemonstratethepossibilityoffasterlong-video
comprehension without substantial loss in performance.
1 Introduction
The frontier for video understanding has moved from short,
curated clips (Mangalam, Akshkulakov, and Malik 2023) to
long-formcontentsuchasfeature-lengthfilms,hour-longlec-
tures,andegocentricrecordings1(Hongetal.2025;Shuetal.
2025),toweeks-longvideosspanningmultipletopics(Rege
etal.2026).Existingworkonansweringaquestionfromsuch
content focused on explicitlocalization(finding the handful
ofsecondsthatmatter)(Zuoetal.2025)andcomprehension
(reasoning over what happens in the localized frames) (Yeo
et al. 2026). Modern VLMs are exceptionally good at pro-
cessing long-context inputs across modalities (Wang et al.
2025c),improvingovercontrastivemethodsthatshowedlit-
tle success (Bagad, Tapaswi, and Snoek 2023).
However, the semantic expertise of VLMs comes at a
tradeoff, and two main paradigms of VLM-based video un-
1https://ego4d-data.org/
parent root
1st children
(annotated at the
beginning)
chosen branch
(L1)
expanded
LLM
explored
path
LLM
chosen
evidence
chosen branch
(L2)
chosen branch
(L3)
unexplored
branch
N1
N2
N3
N4
N5
N6
N7
N8
N9
N10
N11
N12
N13
N14
N15
Initialized nodes
LLM-explored nodes
Nodes not generated in lazy treeFigure 1:Overview of how LazySloth bounds captioning.
Each level captures more fine-grained information, with the
parent root representing all of the video. Grayed-out nodes
are never visited, which reduces the total time to navigate
through the hierarchical tree.
derstanding have emerged. The first places sampled frames
directly into the model’s context and asks for an answer in
oneshot(Shenetal.2025;Shuetal.2025).Thisisbounded
bythecontextwindowandthequadraticcostofattention:an
hour of video sampled at even one frame per second (FPS)
yields thousands of frames and hundreds of thousands of
visual tokens, forcing frame budgets so aggressive that the
evidence needed to answer is often not sampled. To combat
this,asecondparadigmhasemergedthatconvertsthevideo
into text: a VLM captions every frame or every fixed-length
segment, and an LLM then reasons over the resulting cor-
pus(Zuoetal.2025;Yeoetal.2026;Wangetal.2025b).This
circumventsthecontextlimitandyieldsaneasilysearchable
artifact. However, this requires one VLM call per unit of
arXiv:2609.37426v1  [cs.CV]  29 Sep 2026

video, so the cost scales linearly with durationregardless of
thequestionasked.Aquestionaboutafive-secondeventstill
requires captioning the other fifty-nine minutes. Since the
VLM call is much more expensive than any other compo-
nent of the system, this wasted captioning is the bottleneck.
Multimodal retrieval-augmented generation (RAG) ad-
dressespreciselythisissuebyretrievingcandidatesegments
withcontrastiveimage-textencodersbeforeinvokingaVLM
(Radford et al. 2021; Most et al. 2025). Retrieval is cheap,
but the representation is lossy (Yu et al. 2025; Lumer et al.
2026)asthecontrastiveencodercompressesaframeoreven
apoolofframesintoasinglevector.Thisdiscardstemporal,
motion, and fine-grained detail.
Efficient methods retrieve with lossy embeddings while
accuratemethodscaptionextensivelywithVLMs.Fewworks
have asked howVLM-basedretrieval itself should be opti-
mized. To address this, we introduceLazySloth(Figure 2),
a tree-based search method that generates captions lazily.
LazySloth summarizes the video as a top-down tree that is
neverfullyinstantiated.WeshowaLLMarootsummaryof
the whole clip and a handful of coarse branch summaries,
fromwhichitdescendsintothebranchitdeterminesismost
relevanttothequestion.Onlythatbranch’schildrenarecap-
tioned, and we repeat this descent until an interval is small
enoughtoexpandintoper-framedetail.Finally,amultimodal
reasoner answers from the evidence surfaced along the path
together with the key frames the search resolved to. This
boundsthecaptioningcosttothedepthofthedescentrather
thanthelengthofthevideo,andgrowslogarithmicallywhere
prior methods grow linearly.
Across four long-video benchmarks and two recent open-
source VLMs, LazySloth was 2.9–8.3×faster than existing
agentic methods while matching or exceeding both special-
ized long-video VLMs and RAG-based pipelines in accu-
racy. It also narrowed the accuracy gap between the tested
open-source models and a closed-source model, GPT-4o.
Our contributions are:
1.LazySloth,alazyhierarchicalsearchmethodthatbounds
VLM captioning to the branch the search actually ex-
plores, making captioning cost logarithmic rather than
linear in video length.
2. Evidence thatVLM-basedretrieval on frame bunches
(that captures temporal change unlike individual frames)
can be madeefficientwithout falling back on lossy em-
beddings.
3. An evaluation across four long-video benchmarks and
two open base VLMs against specialized video VLMs,
RAGpipelines,andagenticbaselines,showing2.9–8.3×
speedups at comparable or better accuracy.
4. Ablations that separately isolate the contribution of the
hierarchy, of its laziness, and of VLM-based versus
embedding-based scene understanding.2 Related Work
2.1 Vision–Language Models (VLMs)
General Purpose Video VLMsOpen VLM series such
as VideoLLaMA2, Qwen-VL3, InternVL4, LLaVA-
Video/OneVision5, Gemma6, and GLM7are exten-
sively used for building agentic video-understanding sys-
tems.TheseVLMs,however,ingestvideodataassequences
ofuniformlysampledframeswithintheircontextlengthbud-
gets,andrelyonaggressivetruncationordown-samplingfor
longer and high-resolution videos. Although this allows for
gracefulscaling,smalleropenmodelsdegradeafterroughly
30 frames (Wu et al. 2024), consistent with the lost-in-the-
middleeffect(Liuetal.2024).Usingcontext-as-memorydid
notprovideusablememory(seeexperimentsinAppendixB),
whichmotivatedoursearchforefficientretrievalsystemsthat
support hundreds of frames.
Specialized VLMs for Long VideosSpecialized VLMs
have also emerged to address the unique problems of long
video understanding. VideoLLaMA 3 (Zhang et al. 2025), a
well-knownmodelinthisspace,utilizeshigh-qualityvision-
centricimage-textdatasetsinitstrainingcorpusandachieved
2-3% gains on benchmarks like Video Multi-Modal Evalu-
ation (Video-MME) (Fu et al. 2025) and Multi-task Long
VideoUnderstanding(MLVU)(Zhouetal.2025).Otherno-
table models include VideoChat-Flash (Li et al. 2025a) and
video-SALMONN 2+ (Tang et al. 2026), which employs
model architecture and training data innovations to improve
videounderstanding.Morerecently,VideoChat-R1(Lietal.
2025b)integratesspatio-temporalspecificrewardswithrein-
forcementlearningtoimprovetemporalgroundingofVLMs.
Thecorelimitationsofframesamplingandutilizingmemory
as context, however, remain (Section 4).
2.2 Long Video-Understanding Datasets and
Benchmarks
Several high-quality long video-understanding datasets and
benchmarks have been released to evaluate foundation
model capabilities and train small-scale specialized VLMs.
LongVideoBench(Wuetal.2024)andVideo-MME(Fuetal.
2025) are widely used for model evaluation in a multiple-
choice question (MCQ) format, with videos sourced from
web platforms such as YouTube8. More recently, evalua-
tion has focused on extremely long (weeks-long, rather than
hours) video understanding in VLMs (Wang et al. 2025a).
TheEgo4DandEgoLife9corporahavealsobeenutilizedfor
testinglong-formategocentriccontent,withdatasetssuchas
EgoSchema(Mangalam,Akshkulakov,andMalik2023)and
EgoMemReason (Wang et al. 2026) testing short and week-
2https://github.com/damo-nlp-sg/videollama3
3https://github.com/qwenlm/qwen3-vl
4https://github.com/OpenGVLab/InternVL
5https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2
6https://deepmind.google/models/gemma/gemma-4/
7https://z.ai/blog/glm-5
8https://www.youtube.com
9https://ego4d-data.org/, https://egolife-ai.github.io/

Video Sequence
Corpus Creation
VLM Search Agent
DCI Exec
Engine
DCI Tree Search
Hierarchical
TreeRelevant VQA-ed text corpus
Final Output
Label
Frames List
N s
N/2 s
N/4 s
N/8 s
Hierarchical Frame Bunching
Frames List
Lazy Tree Generation
VLM Captioner
Frame Bunch
Tree Node
VLM Predictor
Extracted
Frames
Frame 
Sampling
Frames ListDCI
ShellCommand
Output
loopedbacktrackingFigure 2:Overview of the LazySloth pipeline. (A)Corpus creation, which produced an ordered list of frames after applying
optionaluniformframesamplingtotheinputvideo.(B)Hierarchicalframebunching,whichinitializedthefirstfewlevelsofthe
searchtree.(C)LazygenerationoftreenodesbythecaptionerVLMC.(D)AgenticDirectCorpusInteraction(DCI),wherethe
searchVLMSexploredthetreewithbacktrackingtotraversealternativebranches.(E)Finalanswerpredictionbythereasoner
VLM (R), conditioned on the textual evidence and representative frames collected during the search.
long comprehension, respectively, in MCQ format. Bench-
marks studied in this paper are summarized in Table 1.
2.3 Methods for Long Video-Understanding
RAG-Based MethodsMultimodal retrieval-augmented
generation (RAG) has been explored for long-video under-
standing.Yeoetal.(2026)studiedRAG-basedmethodssuch
as the text-based LightRAG (Guo et al. 2025) and Hip-
poRAG (Gutierrez et al. 2024), and the multimodal Video-
RAG (Luo et al. 2025). Our method resolves the concern of
relyingonlossymultimodalembeddingsbyutilizingseman-
tic LLM-based scene understanding (Section 4.3).
Agentic methodsSeveral agentic methods have been pro-
posed in the literature to improve VLM-based long video
comprehension techniques. Some notable recent work in-
cludes VideoLucy (Zuo et al. 2025), which evaluated the
effectiveness of coarse-to-fine-grained memory-based re-
trieval.Otherrelatedworkhasevaluatedtheeffectofstoring
specialized memory banks for episodic, semantic, and vi-
sual data separately (Yeo et al. 2026), while Wang et al.
(2025b) uses self-reflection-based iterative searching with
VLMs. However, these works did not evaluate how VLM-
basedlongvideosearchcanbeoptimizedwithoutsignificant
performance reductions.
3 LazySloth
Ourproposedmethod,LazySloth(Figure2),answersaquery
Qover a long video without captioning every frame. It uti-
lizes three VLMs: a captioner model,C, that builds a top-
down caption tree lazily, a searcher VLMS, and a reasoner
VLMR.Srepeatedly navigates into a branch that it deter-
minestobemostrelevant.Rfinallyanswersthequerygiventhe selected caption corpus (temporal context) plus the key
framesoftheregionthesearchnarrowedto(visualevidence).
After decoding the video clip for frame extraction, we
get an ordered frame listF=⟨f 1. . . fN⟩(timestamps are
providedincontext).Thetop-downlazytreeistheninitiated,
spanning allNframes, which we caption withCon≤m b
uniformly sampled representative frames. At every step, the
rootissplitintok(branchfactor)roughlyequalchildren,each
of which represents a contiguous spans of the video. Each
span is captioned the same way withC. We initialize the
evidence corpus as the root caption plus the child captions,
and the searcherS’s selected key-frame rangecurrPosis
set to the whole clip[0, N)at timestampt. Each node is
assigned anid.
Atext-onlytop-downsearchthenfollows,duringwhichS
sees the question, an action menu, and a transcript showing
the root and current-level captions. Here, searcherSmust
output exactly one action from the following list:
1.DRILL <id>:Thisactionisusedtodescendintoafron-
tier branch. A node is split into≤kchildren, which are
thencaptionedlazilybyC.Forthesiblingnodesnotcho-
senbyS,LazySlothneverrunscaptioning,whichbounds
thecost.Thenewlygeneratedchildrenarenowthefrontier
of the tree,currPosis updated, and the new captions
are appended to the evidence corpus. We do not discard
thepreviousfrontier;instead,weretainitintheevidence
corpus soScanDRILLa siblingidto backtrack one
level.
2.EXPAND <id>:Ifabranchalreadyhas≤ℓframes(i.e,
itisatleafsize),theVLMcanchoosetoEXPANDinstead.
This step captions every frame in that leaf range episod-
ically, which is then appended to the evidence corpus.
Finally,currPosis set to that leaf.

Algorithm 1: Hierarchical lazy tree generation and retrieval
Require:VideoV,queryQ,labelsL;captionerC,searcher
S, reasonerR; branch factork, leaf sizeℓ, key-
frame budgetm r
Ensure:answerˆy
1:F←SampleFrames(V);N← |F|
2:Ccaptions root[0, N)and itskchildren
3:F ←children
4:Fprev←∅
5:E ← {root, child captions}
6:[a, b)←[0, N)
7:whileround budget remainsdo▷text-only tree search,
budget based on backtracking limit
8: SreadsQand the frontier captions{captionC:c∈
F}and emits ONE action:
9:ifDrillCwithc∈ F ∪F prevandb c−ac> ℓthen
▷ c∈ F prev⇒backtrack one level
10: splitCintokchildren;Ccaptionsonlythem
▷unchosen siblings never captioned
11: Fprev←frontier ofC
12: F ←children
13: [a, b)←[a c, bc)
14: E+=their captions
15:elseifExpandCwithc∈ F∪F prevandb c−ac≤ℓ
then
16: C captions each frame ofC
17: [a, b)←[a c, bc)
18: E+=per-frame captions
19:else ifAnswery Sthen▷ y S:searcher’sprovisional
answer
20: break
21:else▷ifSoutputsaninvalidid,Drillsonaleafnode,
orExpands on a non-leaf
22: promptSto retry
23:K←evenly sample≤m rframes ofF[a:b)in time
order▷key frames of the searcher-narrowed region
24:y R←R(Q, L,E, K)▷multimodal:E= temporal con-
text,K= visual evidence
25:returnˆy←y RifyR̸=Insufficientelsey S
3.PROVISIONAL ANSWER <Q>: The searcherS’s an-
swer toQ, based on the collected evidence, which ends
the search loop.
Finally, both the textual evidence collected from the cap-
tioned corpus by the searcherSand≤m rrepresentative
frames in the leaf nodes of the hierarchical tree thatSnav-
igated down to are passed to the multimodal reasonerR,
whichthenreasonsonaper-framebasisandemitsanoutput
labeloropen-endedtextbasedonthequery.Theimpactofa
post-hoc multimodal reasoner is ablated in Table 4.
Letd=⌈logk(N/ℓ)⌉denote the number of levels navi-
gated until a leaf of at mostℓframes is reached. Since each
navigationcaptionsatmostknewnodesandthefinalexpan-
sioncaptionsatmostℓframes,thetotalnumberofcaptioning
operations is bounded byVideo-
MMELongVideo
BenchLV
BenchEgo
Schema
Source YouTube Web YouTube Ego4D
Domain 6 domains Multi-topic Longform Egocentric
# Videos 900 753 103 500
# Questions 2,700 1,337 1,549 500
# Options 4 4–5 4 5
Avg. length (min) 17.0 7.9 67.3 3.0
Range (min) 0.2–59.7 0.1–59.6 30.1–139.9 3.0 fixed
Total duration (h) 255.3 99.8 115.5 25.0
Table1:Long-videocomprehensionbenchmarksusedinour
evaluation: Video-MME (Fu et al. 2025), LongVideoBench
(Wu et al. 2024), LVBench (Wang et al. 2025a), and
EgoSchema(Mangalam,Akshkulakov,andMalik2023).All
fouraremultiple-choiceandscoredbyexactmatch.Thesuite
spansweb,film/television,andegocentriccontent,andinto-
tal, we evaluate almost 500 hours of video.
C ≤dk+ℓ=k
logkN
ℓ
+ℓ=O(klogkN+ℓ).(1)
For a fixed branching factorkand leaf sizeℓ, this simpli-
fiestoO(klogkN),comparedtotheO(N)frame-captioning
cost incurred by dense captioning. We summarized LazyS-
loth’s optimized video search in Algorithm 1.
3.1 Evaluation Benchmarks
We evaluated the proposed framework on four established
MCQ video question-answering benchmarks with well-
maintainedleaderboards,spanningmovieclips,generalweb
videos, long-form videos, and egocentric videos, following
the discussion in Section 2.2: Video-MME (Fu et al. 2025),
LongVideoBench (Wu et al. 2024), LVBench (Wang et al.
2025a),andEgoSchema(Mangalam,Akshkulakov,andMa-
lik 2023). Table 1 summarizes each dataset’s statistics. For
allofthesebenchmarks,weconductedafullrunandreported
task accuracy as an exact match of the LLM’s output MCQ
label to the ground truth.
3.2 Baselines
For each dataset, we first quantified random baseline perfor-
mance as1/N, whereNis the number of MCQ options for
Video-MME,LongVideoBench,LVBench,andEgoSchema.
We compared LazySloth to multiple classes of methods
starting with general video-understanding models, specifi-
callyQwen3.627B10andGemma431B11,usingatmost64
frames sampled from each video to stay within their context
windows (Yin et al. 2024; Fu et al. 2025; Doorenbos, Spu-
rio, and Gall 2025). Next, specialized video-understanding
models such as VideoChat-R1 (Li et al. 2025b) and Vide-
oLLaMA 3 (Zhang et al. 2025) were tested. To understand
howourmethodcomparestoRAG-basedmethods,wecom-
pared it against the text-based LightRAG (Guo et al. 2025)
10https://huggingface.co/Qwen/Qwen3.6-27B
11https://huggingface.co/google/gemma-4-31B-it

followed by the multimodal Video-RAG (Luo et al. 2025).
Finally,amongmodernagenticbaselines,weincludedVide-
oLucy (Zuo et al. 2025) and WorldMM (Yeo et al. 2026).
3.3 Ablations
We designed two ablations to investigate (a) the behavior of
lazy tree generation, (b) the impact of the tree data struc-
ture, and (c) the effectiveness of using VLMs instead of
multimodal embeddings. We replaced the searcher VLMS
withembedding-basedretrievalbyscoringthetreenodesus-
ing Contrastive Language–Image Pretraining Score (CLIP-
Score) (Hessel et al. 2021). This enabled greedy best-first
search (Greedy BFS) over the tree.
ThesearcherSfirstnavigatedthattreelazilywithDRILL
andEXPAND, and generated its answer using the collected
evidence. This allowed us to illustrate the impact ofVLM-
based scene understanding. Next, we ablated the effect of
lazytreegenerationbyeagerlyconstructingthefullhierarchy
with CLIPScore-based nodes. The searcherSthen received
thefulltreeascontextwhilecollectingevidence.Wekeptall
othercomponents,includingthebacktrackingfromthemain
method, intact, ablating thelazy, on-demand (top-down,
implicitly pruned)tree construction of LazySloth. These
comparisons allowed us to illustrate the efficiency gains of
Algorithm 1 over naive brute-forced methods.
3.4 Experimental Setup & Models Tested
All experiments on existing agentic methods and LazySloth
wererepeatedontwobasesearcherSandreasonerRVLMs
(Qwen3.6 27B and Gemma 4 31B). We ran Qwen 3.6 and
Gemma4’scompiledGGMLUniversalFile(GGUF)locally
through LM Studio12. The captioner modelCwas set to
Nvidia’sNemotron3NanoOmni30Bforanequitablecom-
parisonofeachmethod’smerit(whichwaschosenbasedon
caption quality experiments outlined in Appendix C). We
loaded Nemotron 3’s compiled GGUF locally through LM
Studio. All experiments were run on four Nvidia H100 80
GB, one RTX Pro 6000 96 GB, and two RTX 3090 24 GB
GPUs.Allvideosweresampledat1FPSandmodelsinitial-
ized with a context window of 79,104 tokens to fit on our
available hardware.
4 Results
Table 2 shows the accuracy of LazySloth compared to ex-
isting models and methods. Our method outperformed spe-
cialized models and retrieval-based methods. On Qwen 3.6
27B, LazySloth exceeded the strongest specialized video-
understandingmodeloneverybenchmark:by5.2pointsover
VideoLLaMA 3 on Video-MME, 2.2 on LongVideoBench,
1.5 over Time-R1 on LVBench, and 13.3 over VideoL-
LaMA 3 on EgoSchema. Importantly, these gains were re-
alized with a general-purpose base model (i.e., backbone),
andnovideo-specifictraining.LazySlothalsooutperformed
the best RAG-based baseline (Video-RAG 7B) by at least
6.0pointsonthetwobenchmarkswheretheirnumberswere
12https://lmstudio.ai/Accuracy (%)
MethodVideo-MME
(w/o subs)LongVideo
BenchLV
BenchEgo
Schema
Random baseline 25.0 21.3 25.0 20.0
Direct inference (64 frames)
GPT-4o‡71.9 66.7 48.9 72.2
Qwen 3.6 27B 45.7 31.8 27.7 38.4
Gemma 4 31B 65.8 52.8 40.7 65.4
Specialized video models
VideoChat-R1.5 52.5 49.8 33.4 41.6
VideoChat-Flash†55.6 — 33.2 44.1
VideoLLaMA 3 56.2 51.0 35.4 60.6
Time-R1†— — 37.6 31.1
RAG-based
LightRAG†46.6 — 30.4 —
Video-RAG 7B†55.4 — 33.1 —
Agentic
VideoLucy (Qwen) 34.3 43.9 35.4 51.1
VideoLucy (Gemma)71.3 65.7 53.870.9
WorldMM (Qwen) — — 29.5 —
LazySloth(Qwen) 61.4 53.2 39.1 73.9
LazySloth(Gemma) 66.5 59.5 45.9 69.1
Table 2:Accuracy results on four long video-
understanding benchmarks. Boldmarks the best open-
source result per benchmark; underline marks the second-
best open-source result.†Numbers sourced from Yeo et al.
(2026).‡GPT-4onumbersaretakenfromthepublicleader-
boards of the respective benchmarks; we do not report a
newer GPT release, as none could be located on the main-
tained leaderboards.
available, which was consistent with our claim that embed-
ding retrieval discards fine-grained detail required for so-
phisticated video understanding.
Among other agentic methods such as VideoLucy, using
Qwen3.6 27B as the backbone, we saw that LazySloth im-
proved over VideoLucy by 27.1, 9.3, 3.7, and 22.8 points,
and over WorldMM by 9.6 points on LVBench. The mar-
ginoverdirectinferencewas11.4–35.5points.Wenotethat
VideoLucyunderperformedcomparedtodirectinferenceon
Video-MME with the same backbone (34.3% vs. 45.7%),
whereas LazySloth improved on it by 15.7%.
Improvements were markedly smaller on Gemma 4 31B,
where LazySloth trailed VideoLucy on all four benchmarks
by1.8–7.9points,whichdemonstratedthattheimprovement
margin was largely dependent on the specific model: on av-
erage, Qwen3.6 27B gained 21.0% in accuracy with our
pipeline compared to 4.1% for Gemma 4 31B over direct
inference. Further experiments with more backbone VLMs
would confirm this pattern as a claim. Despite lower perfor-
mance gains with Gemma 4, LazySloth achieved compara-
bleaccuraciesat2.9–8.3×lowermedianwall-clocktimeper
video than competing methods, as we discuss next.
4.1 Efficiency Analysis

2×1045×104Brute-forced eager tree
WorldMMVideoLucy
LazySloth (ours)Direct inference
40 60 80 100 120 140
Video length  (minutes)102103
wall-clock time per video   (seconds, log scale)Figure3:EfficiencycomparisonofLazySloth,abrute-forced
eagertree,VideoLucy,andWorldMMbymediansecondsper
video (log y, broken above the working band) on LVBench,
which was chosen as it has the longest videos in our test
corpus. Bootstrapped 95% confidence intervals are plotted
ateachvideolengthvalue.AllmethodssharedtheQwen3.6-
27B reasoner and Nemotron captioner.
TobetterunderstandthegainsrealizedwithLazySloth,we
firstcomparedthemedianwallclocktimeacrossourmethod
and the two agentic baselines, VideoLucy and WorldMM.
As a control, we included the brute-forced eager bottom-up
tree (Section 3.3). For these tests, we focused only on the
LVBench dataset as it contained the longest videos in our
test set (Table 1). Figure 3 shows that our method was the
fastest,withmediantimetoanswerasingleLVBenchquery
at 194 seconds (secs) compared to 611 secs on VideoLucy,
and 1,139 secs on WorldMM. On the longest videos (140
minuteslong),ourmethodwas3×fasterthanVideoLucyand
7.7×faster than WorldMM. The gap with the brute-forced
eager bottom-up tree remained disproportionately high at
287×slower than LazySloth. Direct inference, which feeds
64framesintotheVLM’smemory,stayedconstantat57secs,
and resulted in notably lower task accuracies. Additional
discussion about efficiency can be found in Appendix D.
Next, we decomposed each method into three categories:
time spent captioning, time spent reasoning, and over-
head (e.g., video decode, I/O, etc). Figure 4 shows the re-
sults on LVBench. We observed that reasoning dominated
WorldMM’smedianwallclocktime(811s),asitusedtriple-
extraction with VLMs across episodic, semantic, and visual
memories.VideoLucy,ontheotherhand,wasdominatedby
its multi-frame coarse-window captions (67%) as compared
to reasoning. Our method optimized both frontiers, and nei-
thercaptioningnorreasoningdominated,ensuring2.9–8.3×
faster median wall clock time than the baselines.
4.2 Behavioral Analysis
We next characterized how LazySloth’s search behaves and
what its behavior predicts about the final accuracy. Figure 5
plots the accuracy difference between LazySloth and direct
inference against the depth the search reached. On Qwen3.6
27B,wenoticedpositivedifferencesacrossalltestedbench-
0 200 400 600 800 1000 1200
Median wall-clock time per video  (seconds)LazySloth (ours)VideoLucyWorldMM 174 811 153 1138 s
409 120 611 s
105 194 sCaptioning Reasoning (LLM) Other (I/O, decode)Figure4:WorldMMvsVideoLucyvsLazySlothmedianwall
clocktimepervideoonLVBenchdecompositionbycaption-
ing,reasoning,andother(I/O,etc.).VideoLucyspendsalot
oftimecaptioning;WorldMMspendsalotonreasoning.Our
method optimizes both frontiers for the lowest total time.
0 2-3 4+
Depth0.00.20.4Accuracy Difference
(LazySloth − direct inference)
Dataset
Video-MME
LongVideoBenchLVBench
EgoSchema
Model
Qwen 3.6 27B Gemma 4 31B
Figure5:ComparingtheaccuracydifferencebetweenLazyS-
lothanddirectinferenceacrossbenchmarksandmodelswith
respect to search tree depth.
marks, while on Gemma 4 31B, we noticed the differences
were zero throughout. This was consistent with the model-
based performance differences in Section 4. Multi-level tree
search did not impact Gemma 4 31B’s final accuracy, and
likely negatively hurt its long video capabilities.
Next, Figure 6 analyzes the performance of LazySloth
with respect to the caption budget during search on Qwen
3.6 27B. The first descent into the tree (moving from five
caption calls to nine) clearly helped the VLM, as accu-
racy rose by 25%. Beyond that, accuracy fell on Video-
MME,LongVideoBench,andLVBench.OnEgoSchema,ac-
curacydeclinedmonotonically,likelyasvideosinthedataset
wereextremelyshort(threeminuteseach).Thisdeclinewas
largely correlated with harder questions, where the search
failed to resolve the query cheaply in higher levels of the
tree.
WeadditionallyinvestigatedLazySloth’sabilitytofindthe
exactspaninthevideothatcontainstheanswertothequery.
Comparedtorandomlychoosingsimilar-sizedwindows,Fig-
ure 7 shows each VLM significantly improved (1.8−2.1×)
at identifying the correct window (Qwen3.6 27Bn= 163
and Gemma 4 31Bn= 171). However, while the choice

45 9 13 17 21
Caption calls implied by the search0.250.500.751.00Accuracy
Video-MME
LongVideoBenchLVBench
EgoSchemaFigure 6: Accuracy with respect to the caption budget used
by the tree search with Qwen3.6 27B, wherebudget= 1
root+ 4seed branches+ 4per successfulDRILL+ 4per
EXPAND. Beyond a single descent into the tree, accuracy
declined across all benchmarks.
Qwen 3.6 27B Gemma 4 31B0.000.050.100.150.200.25P(Final window covers
the ground truth span)1.8× chance2.1× chancecovered the ground truth span
same-width window at random
Figure 7: Analysis of temporal localization on correctly an-
sweredLongVideoBenchquestionsthatwereannotatedwith
specifictimestamps.Blue:theportionwhereLazySloth’sfi-
nal window contained the referenced span. Gray: the rate
expected from placing a window of the same width at ran-
dom, averaged per question.
wasnotarbitrary,thefinalwindowdidnotcontaintheactual
span for86−91%of the questions, where the model still
answered the question correctly.
4.3 Ablation Results
We replaced VLM captioning with a frozen CLIP encoder,
andfoundthataccuracyfellby8.8–19.9%withCLIPScored
nodes compared to a Qwen3.6 27B base model (Table 3).
VLM-basedsceneunderstandingwasamajorcontributorto
LazySloth’s performance. Holding the scorer fixed at CLIP-
Score, the lazy top-down and eager bottom-up construc-
tion differed by at most3.1% with no consistent direction
(Video-MME41.5vs.43.4,LongVideoBench40.5vs.42.1,
EgoSchema55.6vs.56.4). Lazy expansion largely did not
affect accuracy, explaining how LazySloth speeds up video
retrieval (Figure 3) without a massive performance loss.
Finally, we explored the effect of having a separate rea-
sonerVLMinLazySloth,andfoundthatthesearchtrajectory
largely determined the final answer (Table 4). The reasoner
improved micro-averaged accuracy by0.3%, with no clearAccuracy (%)
Configuration Video-MME LongVideoBench LVBench EgoSchema
LazySloth61.4 53.2 39.1 73.9
Scene understanding (CLIPScore, greedy best-first)
lazy top-down tree 41.5 40.5 27.2 55.6
eager bottom-up tree 43.4 42.1 30.3 56.4
Table 3: Ablation study of LazySloth with Qwen3.6 27B.
Group(a)variesthememorystructurewhilekeepingVLM-
basedsceneunderstanding;group(b)replacesVLMcaption-
ing and search with a frozen CLIP encoder that scores each
tree node by CLIPScore and navigates by greedy BFS.
Dataset Searcher (%) Reasoner (%)∆(Reasoner−Searcher)
EgoSchema 76.6 76.9 +0.3
LongVideoBench 58.6 60.8 +2.2
LVBench 42.5 41.4−1.1
Video-MME 62.3 62.6 +0.2
MCQ Micro-Avg 58.0 58.3 +0.3
Table 4: Agreement between the searcher’s selected answer
and the reasoner’s final answer with Qwen 3.6 27B. We ob-
servethatusingaseparatereasonerresultsina0.3%accuracy
gain over always trusting the search model’s answer.
directionality. The searcher’s provisional answer was thus
very nearly as accurate as the final one, and removing the
reasoner’s VLM call could improve efficiency further.
5 Conclusion
We introduce LazySloth, a tree-based search method for
long-video understanding that bounds captioning to regions
a VLM deems relevant rather than exhaustive captioning.
Acrossfourbenchmarksandtwoopen-sourceVLMs,LazyS-
lothwas2.9–8.3×fasterthanexistingmethodswhilematch-
ingorexceedingspecializedvideo-understandingVLMsand
RAG-based methods in accuracy. Detailed failure analysis
showed gaps in VLM behavior where it navigated to the
wrong window or degraded in performance with extra cap-
tion calls. Further, our ablations show LazySloth remained
competitive due to VLM-based scene understanding, and
lazyconstructionwascloseinaccuracywitheagerconstruc-
tion,therebyresultinginamethodthatspeedsuplong-video
retrieval without a substantial cost in task performance.
6 Limitations
One of the empirical shortcomings of LazySloth is that we
observed our method assumed that long-video comprehen-
sioncanbereliablylocalizedtosubtrees,asweobservedpoor
backtracking and navigation to other branches. Our method
alsoreliedontheunderlyingcaptionqualitygeneratedacross
allresolutions,whichwequantifyinAppendixC.Addition-
ally, our tests remain limited to two base models, without
uncertainty quantification with multiple runs.

References
Bagad, P.; Tapaswi, M.; and Snoek, C. G. 2023. Test of
Time: Instilling Video-Language Models with a Sense of
Time . In2023 IEEE/CVF Conference on Computer Vision
andPatternRecognition(CVPR),2503–2516.LosAlamitos,
CA, USA: IEEE Computer Society.
Doorenbos, L.; Spurio, F.; and Gall, J. 2025. Video Panels
for Long Video Understanding.
El Haddad, K.; Bohy, H.; and Dutoit, T. 2025. The Interac-
tion Behavior Dataset: A Dataset of Smiles and Laughs in
Dyadic Interaction. In2025 IEEE International Conference
on Artificial Intelligence and eXtended and Virtual Reality
(AIxVR), 399–403.
Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang,
Z.; Zhou, C.; Shen, Y.; Zhang, M.; Chen, P.; Li, Y.; Lin, S.;
Zhao, S.; Li, K.; Xu, T.; Zheng, X.; Chen, E.; Shan, C.; He,
R.; and Sun, X. 2025. Video-MME: The First-Ever Com-
prehensive Evaluation Benchmark of Multi-modal LLMs in
VideoAnalysis.In2025IEEE/CVFConferenceonComputer
Vision and Pattern Recognition (CVPR), 24108–24118.
Guo, Z.; Xia, L.; Yu, Y.; Ao, T.; and Huang, C. 2025. Ligh-
tRAG:SimpleandFastRetrieval-AugmentedGeneration. In
Christodoulopoulos,C.;Chakraborty,T.;Rose,C.;andPeng,
V., eds.,Findings of the Association for Computational Lin-
guistics:EMNLP2025,10746–10761.Suzhou,China:Asso-
ciation for Computational Linguistics. ISBN 979-8-89176-
335-7.
Gutierrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y.
2024. HippoRAG: Neurobiologically Inspired Long-Term
Memory for Large Language Models. InThe Thirty-eighth
Annual Conference on Neural Information Processing Sys-
tems.
Hessel,J.;Holtzman,A.;Forbes,M.;LeBras,R.;andChoi,
Y. 2021. CLIPScore: A Reference-free Evaluation Metric
for Image Captioning. In Moens, M.-F.; Huang, X.; Specia,
L.; and Yih, S. W.-t., eds.,Proceedings of the 2021 Confer-
enceonEmpiricalMethodsinNaturalLanguageProcessing,
7514–7528. Online and Punta Cana, Dominican Republic:
Association for Computational Linguistics.
Hong, W.; Cheng, Y.; Yang, Z.; Wang, W.; Wang, L.; Gu,
X.; Huang, S.; Dong, Y.; and Tang, J. 2025. MotionBench:
Benchmarking and Improving Fine-grained Video Motion
Understanding for Vision Language Models. InProceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 8450–8460.
Jiang,X.;Zong,Y.;Zheng,W.;Tang,C.;Xia,W.;Lu,C.;and
Liu,J.2020. DFEW:ALarge-ScaleDatabaseforRecogniz-
ingDynamicFacialExpressionsintheWild. InProceedings
of the 28th ACM International Conference on Multimedia,
2881–2889.
Li,X.;Wang,Y.;Yu,J.;Zeng,X.;Zhu,Y.;Huang,H.;Gao,
J.; Li, K.; He, Y.; Wang, C.; Qiao, Y.; Wang, Y.; and Wang,
L. 2025a. VideoChat-Flash: Hierarchical Compression for
Long-Context Video Modeling. arXiv:2501.00574.
Li,X.;Yan,Z.;Meng,D.;Dong,L.;Zeng,X.;He,Y.;Wang,
Y.;Qiao,Y.;Wang,Y.;andWang,L.2025b. VideoChat-R1:Enhancing Spatio-Temporal Perception via Reinforcement
Fine-Tuning. arXiv:2504.06958.
Liu,N.F.;Lin,K.;Hewitt,J.;Paranjape,A.;Bevilacqua,M.;
Petroni, F.; and Liang, P. 2024. Lost in the Middle: How
Language Models Use Long Contexts.Transactions of the
Association for Computational Linguistics, 12: 157–173.
Lumer,E.;Cardenas,A.;Melich,M.;Mason,M.;Dieter,S.;
Subbiah, V. K.; Basavaraju, P. H.; and Hernandez, R. 2026.
Comparison of Text-Based and Image-Based Retrieval in
ModernMultimodalRetrievalAugmentedGenerationLarge
Language Model Systems. InProceedings of the 18th In-
ternational Conference on Agents and Artificial Intelligence
- Volume 5: ICAART, 4619–4626. INSTICC, SciTePress.
ISBN 978-989-758-796-2.
Luo, Y.; Zheng, X.; Li, G.; Yin, S.; Lin, H.; Fu, C.; Huang,
J.; Ji, J.; Chao, F.; Luo, J.; and Ji, R. 2025. Video-RAG:
Visually-alignedRetrieval-AugmentedLongVideoCompre-
hension. InThe Thirty-ninth Annual Conference on Neural
Information Processing Systems.
Mangalam, K.; Akshkulakov, R.; and Malik, J. 2023.
EgoSchema: a diagnostic benchmark for very long-form
video language understanding. InProceedings of the 37th
InternationalConferenceonNeuralInformationProcessing
Systems,NIPS’23.RedHook,NY,USA:CurranAssociates
Inc.
Most, A.; Winjum, J.; Bhattarai, M.; Jones, S.; Ranas-
inghe, N. R.; Biswas, A.; and O’Malley, D. 2025. Lost
in OCR Translation? Vision-Based Approaches to Robust
DocumentRetrieval. InProceedingsofthe2025ACMSym-
posium on Document Engineering, DocEng ’25. New York,
NY, USA: Association for Computing Machinery. ISBN
9798400713514.
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh,
G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark,
J.; Krueger, G.; and Sutskever, I. 2021. Learning Transfer-
ableVisualModelsFromNaturalLanguageSupervision. In
Meila, M.; and Zhang, T., eds.,Proceedings of the 38th In-
ternational Conference on Machine Learning, volume 139
ofProceedings of Machine Learning Research, 8748–8763.
PMLR.
Rege,A.;Sadhu,A.;Li,Y.;Li,K.;Vinayak,R.K.;Chai,Y.;
Lee, Y. J.; and Kim, H. J. 2026. Agentic Very Long Video
Understanding. InLiakata,M.;Moreira,V.P.;Zhang,J.;and
Jurgens, D., eds.,Proceedings of the 64th Annual Meeting
oftheAssociationforComputationalLinguistics(Volume1:
Long Papers), 46575–46602. San Diego, California, United
States: Association for Computational Linguistics. ISBN
979-8-89176-390-6.
Reverdy,J.;O’ConnorRussell,S.;Duquenne,L.;Garaialde,
D.; Cowan, B. R.; and Harte, N. 2022. RoomReader: A
MultimodalCorpusofOnlineMultipartyConversationalIn-
teractions. In Calzolari, N.; Béchet, F.; Blache, P.; Choukri,
K.;Cieri,C.;Declerck,T.;Goggi,S.;Isahara,H.;Maegaard,
B.; Mariani, J.; Mazo, H.; Odijk, J.; and Piperidis, S., eds.,
ProceedingsoftheThirteenthLanguageResourcesandEval-
uationConference,2517–2527.Marseille,France:European
Language Resources Association.

Shen, X.; Xiong, Y.; Zhao, C.; Wu, L.; Chen, J.; Zhu, C.;
Liu, Z.; Xiao, F.; Varadarajan, B.; Bordes, F.; Liu, Z.; Xu,
H.;Kim,H.J.;Soran,B.;Krishnamoorthi,R.;Elhoseiny,M.;
and Chandra, V. 2025. LongVU: Spatiotemporal Adaptive
Compression for Long Video-Language Understanding. In
Forty-second International Conference on Machine Learn-
ing.
Shu, Y.; Liu, Z.; Zhang, P.; Qin, M.; Zhou, J.; Liang, Z.;
Huang, T.; and Zhao, B. 2025. Video-XL: Extra-Long Vi-
sion Language Model for Hour-Scale Video Understanding.
InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 26160–26169.
Siniukov, M.; Yin, Y.; Fast, E.; Qi, Y.; Monga, A.; Kim,
A.; and Soleymani, M. 2024. SEMPI: A Database for Un-
derstanding Social Engagement in Video-Mediated Multi-
party Interaction. InProceedings of the 26th International
Conference on Multimodal Interaction, ICMI ’24, 546–555.
NewYork,NY,USA:AssociationforComputingMachinery.
ISBN 9798400704628.
Tang, C.; Li, Y.; Yang, Y.; Zhuang, J.; Sun, G.; Li, W.;
MA,Z.;andZhang,C.2026. video-SALMONN2:Caption-
Enhanced Audio-Visual Large Language Models.
Vuillecard,P.;Farkhondeh,A.;Villamizar,M.;andOdobez,
J.-M.2024. CCDb-HG:NovelAnnotationsandGaze-Aware
RepresentationsforHeadGestureRecognition.In2024IEEE
18th International Conference on Automatic Face and Ges-
ture Recognition (FG), 1–9.
Wang, W.; He, Z.; Hong, W.; Cheng, Y.; Zhang, X.; Qi, J.;
Ding,M.;Gu,X.;Huang,S.;Xu,B.;Dong,Y.;andTang,J.
2025a. LVBench: An Extreme Long Video Understanding
Benchmark. In2025 IEEE/CVF International Conference
on Computer Vision (ICCV), 22958–22967.
Wang,X.;Zhang,Y.;Zohar,O.;andYeung-Levy,S.2025b.
VideoAgent: Long-Form Video Understanding with Large
LanguageModelasAgent. InLeonardis,A.;Ricci,E.;Roth,
S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds.,Com-
puter Vision – ECCV 2024, 58–76. Cham: Springer Nature
Switzerland. ISBN 978-3-031-72989-8.
Wang, Z.; Yoon, J.; Yu, S.; Islam, M. M.; Bertasius, G.; and
Bansal, M. 2025c. Video-RTS: Rethinking Reinforcement
Learning and Test-Time Scaling for Efficient and Enhanced
Video Reasoning. In Christodoulopoulos, C.; Chakraborty,
T.;Rose,C.;andPeng,V.,eds.,Proceedingsofthe2025Con-
ferenceonEmpiricalMethodsinNaturalLanguageProcess-
ing, 28126–28140. Suzhou, China: Association for Compu-
tational Linguistics. ISBN 979-8-89176-332-6.
Wang, Z.; Zhang, Y.; Yu, S.; Zhang, C.; Zhao, Z.;
Yoon, J.; Lee, H.; Bertasius, G.; and Bansal, M. 2026.
EgoMemReason: A Memory-Driven Reasoning Bench-
mark for Long-Horizon Egocentric Video Understanding.
arXiv:2605.09874.
Wu,H.;Li,D.;Chen,B.;andLi,J.2024. LongVideoBench:
ABenchmarkforLong-contextInterleavedVideo-Language
Understanding. In Globerson, A.; Mackey, L.; Belgrave,
D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C., eds.,
Advances in Neural Information Processing Systems, vol-
ume 37, 28828–28857. Curran Associates, Inc.Yeo, W.; Kim, K.; Yoon, J.; and Hwang, S. J. 2026.
WorldMM: Dynamic Multimodal Memory Agent for Long
Video Reasoning. InProceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition (CVPR),
25599–25609.
Yin, S.; Fu, C.; Zhao, S.; Li, K.; Sun, X.; Xu, T.; and Chen,
E. 2024. A survey on multimodal large language models.
National Science Review, 11(12): nwae403.
Yu, S.; Tang, C.; Xu, B.; Cui, J.; Ran, J.; Yan, Y.; Liu, Z.;
Wang, S.; Han, X.; Liu, Z.; and Sun, M. 2025. VisRAG:
Vision-based Retrieval-augmented Generation on Multi-
modality Documents. InThe Thirteenth International Con-
ference on Learning Representations.
Zhang, B.; Li, K.; Cheng, Z.; Hu, Z.; Yuan, Y.; Chen, G.;
Leng, S.; Jiang, Y.; Zhang, H.; Li, X.; Jin, P.; Zhang, W.;
Wang, F.; Bing, L.; and Zhao, D. 2025. VideoLLaMA
3: Frontier Multimodal Foundation Models for Image and
Video Understanding. arXiv:2501.13106.
Zhou, J.; Shu, Y.; Zhao, B.; Wu, B.; Liang, Z.; Xiao, S.;
Qin, M.; Yang, X.; Xiong, Y.; Zhang, B.; Huang, T.; and
Liu,Z.2025. MLVU:BenchmarkingMulti-taskLongVideo
Understanding.In2025IEEE/CVFConferenceonComputer
Vision and Pattern Recognition (CVPR), 13691–13701.
Zuo,J.;Deng,Y.;Kong,L.;Yang,J.;Jin,R.;Zhang,Y.;Sang,
N.; Pan, L.; Liu, Z.; and Gao, C. 2025. VideoLucy: Deep
Memory Backtracking for Long Video Understanding. In
The Thirty-ninth Annual Conference on Neural Information
Processing Systems.

Appendices
A Symbol Glossary
Table 11 presents a glossary of all formalization symbols
used in Section 3 and Algorithm 1.
B FPS Sweep Results
Table 5 showcases uniform FPS sampling results on a
few short video datasets: DFEW (Jiang et al. 2020), CCDb-
IB (El Haddad, Bohy, and Dutoit 2025), CCDb-HG (Vuille-
cardetal.2024),CCDb+,SEMPI(Siniukovetal.2024),and
RoomReader(Reverdyetal.2022).Wespecificallyselected
themastheyareafewsecondslongandthereforefitentirely
into VLM context windows at high FPS sampling rates.
The sweep showed that adding frames did not add usable
information.Neithermodelimprovedmonotonicallyassam-
pling rose from 1 FPS to every frame. Qwen 2.5 VL 7B13
on RoomReader fell from 47.0% at 1 FPS to 11.5% when
giventhefullframeset,andonSEMPIfrom54.0%to32.7%.
Gemma 4 E4B14, on the other hand, stayed essentially flat
across all four rates.
Sincetheseclipsareshortenoughtofitentirelyincontext,
the ceiling is not the context window but the model’sability
tousewhatisalreadyinit.Thisexperimentfurthermotivated
a retrieval-based design.
C Captioning Quality Tests
Because the captionerCis held frozen across every
method, we tested their sensitivity before committing to
Nvidia’sNemotron3NanoOmni.Table6showsthatacross
six viable candidates, CLIPScore on frames randomly sam-
pled from the six tested long-video datasets spanned only
0.2331 to 0.2447 CLIPScore, a range of roughly 5%. Im-
portantly, only Gemma 4 E4B scored particularly worse at
0.2153. Since modern VLMs are capable of captioning im-
ages, this result is not quite surprising.
This brings us to the proposition of efficiency. To make
LazySloth fast, we needCto balance captioning quality
withlatency.Acrossthesixcandidates,wenotethattheper-
captionlatencyspanned0.59to2.70seconds,demonstrating
a difference of 4.6×from the fastest to the slowest model.
SinceCis invoked once per tree node and therefore sits in
the inner loop of every method we compare, latency is the
dimension that actually separates the candidates. Nemotron
3NanoOmniwasthefastestcaptionerthatwasnotaquality
outlier. Therefore, we chose this model for LazySloth and
across all methods so that no baseline is advantaged or pe-
nalized.Thisallowedustocleanlycomparethemeritofeach
retrieval method.
Moreover, looking at Table 7, we want to discuss
Nemotron’s high CLIPScore despite the fact that it wrote
thesecond-shortestcaptions.AsseeninTable6,CLIPScore
is directly correlated to average caption length (a known
13https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
14https://huggingface.co/google/gemma-4-E4Bproperty of the metric). This makes Nemotron’s per-word
caption quality competitive. Second, the per-dataset spread
withinanymodeliscomparabletothespreadbetweenmod-
els, which means the differences in Table 7 do not account
for a strict ranking.
D Additional Efficiency Analysis of
LazySloth
Forafine-grainedunderstandingoftheefficiencyanalysisin
Section4.1,webrokedownmedianinferencetimeundereach
method and direct inference in Table 8 for the benchmark
with the longest videos, LVBench. We found that inference
time under each retrieval-based method grew linearly with
video length, but the informative quantity is the slope of
growth. While LazySloth paid 2.43 additional seconds per
extra minute of video (s/min), VideoLucy paid 5.53s, and
WorldMM paid 19.86s/min. We note that these differences
were significant, as WorldMM paid 17.43 s/min more than
LazySloth(t= 13.2,p <10−39)andVideoLucy3.10s/min
more (t= 4.9,p <10−5). LazySloth therefore addressed
the scaling problem that these methods struggled with.
This difference, however, stemmed from both the number
of caption calls and how they were structured. As shown in
Table9,LazySloth’scaptioncountwaseffectivelycompared
to WorldMM and the brute-forced baseline, growing only
from 35 on 40-minute-long videos to 45 under 140-minute-
long videos due to theO(klogkN)bound of Equation (1).
WorldMM grew from 97 to 330, and the brute-forced eager
trees from 3,600 to 12,603, a factor of 280 at the longest
videos. VideoLucy, however, is the interesting case as it is-
sued 27 to 54 calls, statistically indistinguishable from ours,
yet still ran roughly three×slower.
ThisisbecauseVideoLucyre-captionsdensefixed-length
overlapping windows, so each call carries more frames and
morevisualtokensthanaLazySlothbunchcaption.Figure4
shows that captioning consumed 67% of its total time. Our
bounded search cost precisely addressed this issue.
E Additional Benchmark-wise Behavioral
Analysis
To better understand the impact of tree depth searched by
the searcherSon final accuracy, we plotted the aggregate
depth trend of Figure 5 by benchmark and backbone in Fig-
ure10.Directinferencehasnosearchandthereforenodepth
of its own; it is scored on exactly the questions that fall
in each of LazySloth’s strata. The decline we plotted in Fig-
ure10againstDirectInferenceindicatedwhetherthedeeper-
searched strata are indeed the harder questions. This helps
useliminateanimportantconfoundthatcouldmisguidethis
discussion. The takeaway, therefore, is the performance dif-
ference between direct inference and LazySloth, which we
plotted with 95% confidence interval bars.
We note that for Qwen3.6 27B, LazySloth consistently
improvedperformancefromdirectinference,withthediffer-
ences reaching statistical significance at higher tree depths
acrossthefourbenchmarks,mostsharplyonEgoSchemaand
LongVideoBench.

Tier Dataset “All” (31FPS) 24FPS 12FPS 1FPS
Qwen 2.5 VL 7B(Unsloth bf16 GGUF)
DFEWC:26.7±22.7
R:12.3±6.7C:27.4±22.4
R:14.7±7.4C:32.2±24.0
R:26.2±13.7C:32.5±22.7
R:30.9±21.814.05±2.51
CCDb-IBC:23.5±10.1
R:18.0±8.4C:28.2±8.4
R:21.7±7.9C:22.6±8.2
R:22.2±7.0C:27.2±8.0
R:14.8±6.712.63±2.39
CCDb-HGC:16.7±22.5
R:7.4±10.0C:16.2±21.5
R:10.1±13.5C:16.8±20.4
R:13.9±12.9C:20.1±23.8
R:17.8±13.716.73±2.74
CCDb+C:54.4±20.7
R:20.1±12.1C:59.8±25.5
R:22.7±12.1C:60.7±25.1
R:30.2±12.2C:56.7±19.9
R:57.5±10.549.87±3.35
SEMPIC:32.7±15.4
R:1.2±2.6C:44.8±18.3
R:3.4±3.9C:43.2±17.7
R:35.4±11.3C:54.0±19.5
R:56.3±15.450.07±3.62
RoomReaderC:11.5±7.0
R:4.0±5.0C:32.8±17.1
R:21.9±10.9C:34.1±21.2
R:29.3±18.0C:47.0±38.5
R:56.0±10.049.86±3.45
Gemma 4 E4B(Unsloth bf16 GGUF)
DFEWC:30.8±26.7
R:36.2±23.1C:31.3±25.9
R:35.6±22.9C:30.9±25.3
R:38.3±25.4C:32.8±25.0
R:38.1±20.314.05±2.51
CCDb-IBC:25.3±11.4
R:26.4±9.6C:26.4±12.3
R:26.6±9.2C:26.1±10.5
R:27.5±9.9C:24.2±9.6
R:26.8±8.512.63±2.39
CCDb-HGC:16.3±23.5
R:21.1±22.7C:18.6±24.2
R:22.8±24.2C:18.0±23.5
R:20.9±23.0C:18.4±23.7
R:16.2±14.216.73±2.74
CCDb+C:63.0±25.7
R:8.1±5.6C:65.8±26.4
R:9.7±7.6C:63.8±23.9
R:10.1±7.1C:63.5±27.0
R:7.4±5.549.87±3.35
SEMPIC:65.0±12.0
R:61.0±12.0C:68.1±9.8
R:67.5±12.9C:68.5±13.2
R:71.3±12.0C:61.2±17.5
R:64.5±12.550.07±3.62
RoomReaderC:36.9±23.8
R:39.6±27.6C:49.0±32.6
R:48.8±32.4C:47.1±37.6
R:47.5±33.1C:48.9±43.5
R:41.6±31.349.86±3.45
Table5:Perquantization×sampling-strategyestimatesofaccuracy(%)forQwen2.5VL7BandGemma4E4B.Eachvideo
cell reports Constrained (C) and Reasoning (R) settings as mean±std; image-based datasets are frame-independent and span
all sampling columns.
ModelCLIP
Score↑SigLIP
Score↑Avg. length
(words)Latency
(s)↓
google/gemma-4-12b0.24470.1367 32.6 1.27
qwen/qwen3.6-27b 0.24350.140331.4 2.29
google/gemma-4-31b 0.2417 0.1348 29.1 2.70
qwen/qwen3.5-9b 0.2417 0.1402 31.5 0.92
qwen2.5-vl-7b-instruct 0.2341 0.1292 27.7 0.98
nvidia/nemotron-3-nano-omni 0.2331 0.1340 25.1 0.59
google/gemma-4-e4b-it 0.2153 0.1175 21.8 0.77
Table 6: Reference-free caption quality of the candidate
frozencaptionersC,pooledoverallevaluateddatasets.Cap-
tionsweregeneratedfromthesamesampledframeswiththe
identicalpipelineprompt,thenscoredbyCLIPScore(Hessel
etal.2021)andaSigLIP2cross-check.Latencyiswall-clock
secondspercaption.Theshadedrowisthecaptionerusedin
all experiments reported in this paper.
On Gemma 4 31B, however, the two curves were
essentially coincident on EgoSchema and LVBench,
and LazySloth directionally improved performance on
LongVideoBench, although not statistically significant.
Sinceboththeblueandorangelinesintheplotrefertoidenti-
calquestionswithineachstratum,thedifferencecannotbeat-ModelEgo
SchemaLongVideo
BenchMMBench
VideoMovie
ChatVideo
MME
google/gemma-4-31b 0.236 0.239 0.246 0.248 0.238
qwen/qwen3.6-27b 0.236 0.238 0.245 0.262 0.237
google/gemma-4-12b 0.227 0.239 0.248 0.264 0.245
qwen/qwen3.5-9b 0.236 0.229 0.244 0.259 0.240
qwen2.5-vl-7b-instruct 0.229 0.230 0.230 0.256 0.226
nvidia/nemotron-3-nano-omni 0.234 0.220 0.230 0.2520.230
google/gemma-4-e4b-it 0.181 0.219 0.220 0.229 0.229
Table7:Per-datasetCLIPScoreforeachcandidatecaptioner.
Rows are ordered by the pooled CLIPScore of Table 6.
tributedtoquestiondifficulty,andinsteadreflectshowmuch
a given base model gains from our method. Gemma 4 31B
consistently struggled to gain any performance across the
board,supportingourclaimthatgainsaremodel-dependent,
and Gemma 4 31B lacks any headroom to improve perfor-
mance with retrieval-based or agentic frameworks.
Further, Figure 9 examines which branch the search de-
scends into first against randomly picking any branch from
thegeneratedchildren.Wenotethatmodelspreferredthefirst
frontiersplitconsistentlymorethantheotherthreebranches,
which have a roughly equal chance of being picked. The

Length LazySloth VideoLucy WorldMM Brute-forced Eager DI
40 min 149 s 509 s (3.4×) 735 s (4.9×) 7.0 h (169×) 57 s
60 min 176 s 553 s (3.1×) 1109 s (6.3×) 10.5 h (214×) 57 s
90 min 251 s 727 s (2.9×) 1560 s (6.2×) 15.8 h (226×) 57 s
120 min 287 s 1000 s (3.5×) 2393 s (8.3×) 21.0 h (263×) 57 s
140 min 307 s 936 s (3.0×) 2371 s (7.7×) 24.5 h (287×) 57 s
Table8:MedianinferencetimecomparisonacrossdifferentvideolengthsonLVBench.SpeedupisshownrelativetoLazySloth.
Length LazySloth VideoLucy WorldMM Brute-forced Eager
40 min 35 27 (0.8×) 97 (2.8×) 3,600 (103×)
60 min 35 35 (1.0×) 149 (4.3×) 5,401 (156×)
90 min 38 46 (1.2×) 220 (5.8×) 8,102 (215×)
120 min 34 54 (1.6×) 291 (8.6×) 10,801 (320×)
140 min 45 42 (0.9×) 330 (7.3×) 12,603 (280×)
Table9:MediancaptioncallsacrossdifferentvideolengthsonLVBench.SpeedupisshownrelativetoLazySlothwithQwen3.6
27B.
3×103104
40 60 80 100 120 140
Video length  (minutes)102
Brute-forced eager tree
WorldMM
VideoLucy
LazySloth (ours)
Direct inference
VLM caption calls per video   (log scale)
Figure8:WorldMMvsVideoLucyvsLazySlothmediancap-
tions generated per-video on LVBench. While VideoLucy
doesn’t caption significantly more than LazySloth, the dif-
ference in wall clock time arises from how the two methods
utilize caching and optimize multi-frame inputs to VLMs.
first descent landed in the opening quarter of the video on
49% of Video-MME and LongVideoBench questions, 54%
of LVBench, and 64% of EgoSchema, which was roughly
twice the random rate.
Finally, Table 10 matched LazySloth against direct infer-
enceonthesamequestionsandsplittheresultbythequestion
type. We note that gains in performance were not uniform.
Question types, mostly those requiring a moment to be lo-
catedandthenreasonedabout,producedthelargestimprove-
ments (LongVideoBench’s T2O gained 44.7%, SOS 38.3%,
and TOS 37.0%, and Video-MME’s Information Synop-
sis and Temporal Reasoning gained 25.5% and 23.2% re-
spectively). This is an architectural improvement LazySloth
1st quarter 2nd 3rd 4th
Position within the video of the first branch drilled0.00.10.20.30.40.50.6Share of questionsbranch picked at random
Video-MME  (n=1314)
LongVideoBench  (n=289)
LVBench  (n=316)
EgoSchema  (n=372)Figure9:ShareofquestionsforwhichLazySloth(aggregated
overQwen3.627BandGemma431B)chosethefirst,second,
third, or fourth quarter of the video at the first level of the
search tree. LazySloth’s first descent was strongly biased
towardthestartofthevideo(roughlytwicetherateofrandom
branchselection).Thedashedreferenceshowsanempirically
measured random baseline.
brings to the table. Types answerable from any representa-
tive frame, as expected, gained almost nothing. Our method
helpsthemostincaseswherefindingtherightmomentisthe
task,whichexplainstheperformancedifferencesreportedin
Table 2.

none 2-3 4+0.000.250.500.751.00Qwen 3.6 27B
Accuracy
n=217n=1,312n=1,168Video-MME
none 2-3 4+n=287n=403n=647LongVideoBench
none 2-3 4+n=87n=294n=1,168LVBench
none 2-3 4+n=61n=439EgoSchema
none 2-3 4+0.000.250.500.751.00Gemma 4 31B
Accuracy
n=265n=1,717n=700
none 2-3 4+n=265n=621n=451
none 2-3 4+n=71n=478n=1,000
none 2-3 4+n=126n=374
Depth is a property of LazySloth, not of Direct Inference
LazySloth (defines the strata) Direct inference, same questions (difficulty gauge)Figure 10: Tree depth searched by the two tested base VLMs, Gemma 4 31B and Qwen3.6 27B, across the four datasets with
LazySloth and Direct Inference. Since Direct Inference does not create trees to search in, it is scored on exactly the questions
that fall in each of LazySloth’s strata.
Benchmark Question typeNLazySloth Direct inf.∆McNemarχ2
LongVideoBench SSS 97 38.1 12.4 +25.8 18.6∗
LongVideoBench E3E 94 52.1 28.7 +23.4 13.8∗
LongVideoBench S2E 93 57.0 43.0 +14.0 4.4∗
LongVideoBench S2A 88 64.8 62.5 +2.3 0.0
LongVideoBench O2E 87 50.6 42.5 +8.0 1.9
LongVideoBench TAA 82 46.3 26.8 +19.5 7.0∗
LongVideoBench SOS 81 61.7 23.5 +38.3 23.1∗
LongVideoBench T2A 79 54.4 51.9 +2.5 0.0
LongVideoBench T2O 76 65.8 21.1 +44.7 28.7∗
LongVideoBench T3O 74 50.0 14.9 +35.1 22.3∗
LongVideoBench TOS 73 47.9 11.0 +37.0 20.5∗
LongVideoBench T3E 73 43.8 24.7 +19.2 7.0∗
LongVideoBench SAA 72 58.3 27.8 +30.6 17.0∗
LongVideoBench S2O 72 61.1 40.3 +20.8 7.3∗
LongVideoBench O3O 66 54.5 28.8 +25.8 8.8∗
LongVideoBench E2O 65 47.7 43.1 +4.6 0.2
LongVideoBench T2E 65 52.3 27.7 +24.6 9.4∗
Video-MME Object Reasoning 453 57.8 36.0 +21.9 55.5∗
Video-MME Object Recognition 354 65.3 60.5 +4.8 2.5
Video-MME Information Synopsis 322 72.4 46.9 +25.5 54.7∗
Video-MME Action Recognition 313 55.0 52.7 +2.2 0.4
Video-MME Action Reasoning 285 54.7 33.3 +21.4 33.0∗
Video-MME Counting Problem 267 37.1 32.2 +4.9 1.6
Video-MME Attribute Perception 222 67.6 63.5 +4.1 1.0
Video-MME Temporal Reasoning 177 42.4 19.2 +23.2 20.8∗
Video-MME OCR Problems 139 56.8 61.2 -4.3 0.6
Video-MME Spatial Reasoning 56 76.8 58.9 +17.9 4.0∗
Video-MME Temporal Perception 55 70.9 61.8 +9.1 0.8
Video-MME Spatial Perception 54 70.4 61.1 +9.3 1.1
Table10:SearchgainbyquestiontypeusingQwen3.627BasthebaseVLM.LVBenchandEgoSchemaareomittedastheydo
not contain segregated question types.∗marksχ2>3.84(p <0.05).

Symbol Meaning
VInput video clip.
QQuery/task (MCQ question + options, open-ended question, or classification prompt).
LLabel set — MCQ letters / class labels;∅for open-ended QA (free-text answer).
CFrozencaptionerVLM;onlyeverproducescaptions(keptfixedsoanexperimentvaries
only the search VLM).
SSearch VLM + reasoner VLM — one model in two roles: text-only navigation, then
multimodal final answer.
ϕFrame sampling rate, measured in FPS, used to decode the clip.
F=⟨f 1, . . . , f N⟩Ordered extracted frames; eachf i= (idx i, ti,pathi)= native index, timestamp, JPEG
path.
NNumber of extracted frames,=|F|.
kBranch factor — children perDrill, and the root’s fan-out.
ℓLeaf size — frame-range threshold; once a node spans≤ℓframes,Drillstops and
Expand(per-frame captions) is allowed.
mb Max representative frames shown toCwhen captioning one node/bunch (evenly sub-
sampled); “b” = bunch.
mr MaxkeyframesshowntothereasonerS(evenlysubsampled,timeorder);“r”=reasoner.
RMax VLM search rounds (round budget).
smax Consecutive no-progress rounds before an answer is forced (= 3).
F(frontier) The currently-shown sibling nodes — the candidates forDrill/Expand.
prev /F prev The previous frontier, retained soScanDrilla sibling to backtrack one level.
E(evidence) The caption corpus surfaced along the path (root→branch→per-frame captions).
[a, b)/currPosThe position range intoFthe traversal narrowed to; the reasoner’s key-frame window
(starts[0, N)).
node.pos= [a c, bc)A node’s half-open position range intoF; its spanb c−acis compared againstℓ.
level / id Node depth (root= 0) / node id (its start frame index — stable, unique within a level).
cap[·]On-diskcaptioncache,keyedbythefrozencaptionerC(semantictreefornodes,episodic
for frames); reused across queries.
yag, yre,ˆyTheVLM’sanswer,thereasoner’sanswer,andthefinalprediction(ˆy=y re,oryagifthe
reasoner returnsInsufficient).
Table 11: Notation reference for formalization of the LazySloth framework in Section 3 and Algorithm 1.