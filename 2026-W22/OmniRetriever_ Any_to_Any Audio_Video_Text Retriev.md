# OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via Fusion-as-Teacher Distillation

**Authors**: Yunze Liu, Chi-Hao Wu, Enmin Zhou, Junxiao Shen

**Published**: 2026-05-26 07:26:23

**PDF URL**: [https://arxiv.org/pdf/2605.26641v1](https://arxiv.org/pdf/2605.26641v1)

## Abstract
Unified multimodal embedding spaces have become the standard interface for cross-modal retrieval and multimodal RAG, and recent audio-video-text (AVT) encoders extend this setting to three modalities. Such encoders can produce a joint (T,V,A) embedding whenever all three modalities are available, but standard pairwise InfoNCE objectives leave this signal unused during training. We close this gap with fusion-as-teacher distillation, which treats a stop-gradient copy of the fused embedding as a teacher signal for the single-modal embeddings, paired with a Tuple-InfoNCE term that supervises the fused embedding directly. We instantiate this objective as OmniRetriever-7B. Across six zero-shot retrieval benchmarks, OmniRetriever-7B surpasses the closed-source Gemini Embedding 2 by 13.3-18.0 R@1 on Clotho and SoundDescs, and reaches the contemporary zero-shot specialist band of open video-text encoders on MSR-VTT and MSVD. To stress-test joint representations, we further release OmniRetriever-Bench, a 12-direction AVT retrieval benchmark totaling 3782 triples; on it OmniRetriever-7B attains AVG-all 34.84, improving over Gemini Embedding 2 by 1.72 and over the best prior open-source AVT method by 8.03.

## Full Text


<!-- PDF content starts -->

OmniRetriever: Any-to-Any Audio-Video-Text Retrieval
via Fusion-as-Teacher Distillation
Yunze Liu Chi-Hao Wu Enmin Zhou Junxiao Shen
Memories.ai Research
Project Page: OmniRetriever
Abstract
Unified multimodal embedding spaces have be-
come the standard interface for cross-modal re-
trieval and multimodal RAG, and recent audio-
video-text (A VT) encoders extend this setting
to three modalities. Such encoders can pro-
duce joint (T, V, A) embedding whenever all
three modalities are available, but standard pair-
wise InfoNCE objectives leave this signal un-
used during training. We close this gap with
fusion-as-teacher distillation, which treats
a stop-gradient copy of the fused embedding
as a teacher signal for the single-modal em-
bedding, paired with aTuple-InfoNCEterm
that supervises the fused embedding directly.
We instantiate this objective asOmniRetriever-
7B. Across six zero-shot retrieval benchmarks,
OmniRetriever-7Bsurpasses the closed-source
Gemini Embedding 2 by 13.3–18.0 R@1 on
Clotho and SoundDescs, and reaches the con-
temporary zero-shot specialist band of open
video–text encoders on MSR-VTT and MSVD.
To stress-test joint representations, we further
releaseOmniRetriever-Bench, a 12-direction
A VT retrieval benchmark totaling 3,782 triples;
on itOmniRetriever-7Battains A VG-all 34.84 ,
improving over Gemini Embedding 2 by +1.72
and over the best prior open-source A VT
method by +8.03 . Model weights, datasets,
and code will be released.
1 Introduction
Cross-modal retrieval relies on encoders that map
queryable modalities into a single shared embed-
ding space. The canonical CLIP recipe (Rad-
ford et al., 2021) trains two separate towers and
aligns them with pairwise InfoNCE. Recent omni-
modal systems extend this to three modalities of
{T, V, A} and split into two architectural families.
Multi-encodersystems such as ImageBind (Gird-
har et al., 2023) and LanguageBind (Zhu et al.,
2024) keep separate per-modality encoders and
align them pairwise to a fixed image or languageanchor.Unifiedencoders such as WA VE (Tang
et al., 2025) and Omni-Embed-Nemotron (Xu et al.,
2025) instead route all three modalities through one
backbone and produce a joint embedding zTV A on
a single joint forward pass.
Unified encoders are particularly attractive for
A VT retrieval because zTV A is exactly the repre-
sentation that a dual-modal or full-tuple query (e.g.,
T+V→A orA+T+V→T ) needs at inference.
Yet the standard training recipe for unified A VT en-
coders never invokes this joint forward as a super-
vision signal: WA VE and Omni-Embed-Nemotron
both optimise three pairwise InfoNCE losses on
the single-modal sub-encoders only, and the joint
output is computed only at inference time.
The result is that the single-modal sub-encoders
of current unified A VT systems are trained in isola-
tion, with no signal about their cross-modal neigh-
bours. Empirically, this gap is sharpest on audio-
anchored retrieval, where the modality-to-text co-
occurrence is the weakest in standard training data.
On Clotho A →T, closed Gemini Embedding 2
reaches R@1 = 1.34 and the best open omni-
modal system, Omni-Embed-Nemotron, reaches
3.5, while CLAP-family audio–text specialists (Wu
et al., 2023; Mei et al., 2024; Niizumi et al., 2025)
trained on (T, A) pairs alone reach 25to26on the
same direction. The same mechanism limits all
twelve any-to-any directions over {T, V, A} (six
single-modal, six dual-modal) that a practical A VT
retriever has to serve.
We close this gap by using the joint embed-
ding itself as the supervision signal.Fusion-as-
teacher distillation LDtakes a stop-gradient copy
ofzTV A as the teacher for the single-modal sub-
encoders: each of zT,zV,zAis pulled toward
zTV A by InfoNCE. Because the teacher is the same
backbone consumed jointly rather than an exter-
nal encoder, the sub-encoders receive the cross-
modal context that no unimodal teacher provides,
at the cost of one additional joint forward pass
1arXiv:2605.26641v1  [cs.CV]  26 May 2026

(a)Prior open Omni-Modality Embedders
WAVE · ImageBind · LanguageBind · Omni-Nemotron  —  pairwise InfoNCE only
T text
V video
A audiof θ
shared
encoderzT
zV
zAL A
pairwise
InfoNCE
zTVAemergent byproduct
✗ unused in any loss(b)OmniRetriever (ours)
fusion-as-teacher distillation  +  Tuple-InfoNCE on the joint embedding
T text
V video
A audiof θ
shared
encoderzT
zV
zA+ LA pairwise (kept)
zTVAL T     T u p l e - I n f o N C E
sharpens zTVA directlyL D     D i s t i l l a t i o n
sg(z_TVA) →  z_T, z_V, z_A
(c)12-direction Omni-Modality Retrieval
First open 7B-scale embedder to surpass closed Gemini Embedding 2
0 10 20 30 40  R@1WAVE-7B
open · 7B25.32
Omni-Nemotron
open · 3B26.81
Gemini Emb. 2
closed · API33.12
OmniRetriever
ours · open · 7B34.84 +1.72(d)External audio–text benchmarks
Closes the audio gap by +13.3 to +18.0 R@1 — enters the specialist SOTA band
SPECIALIST SOTA BAND
0 5 10 15 20 25 30  R@1
Gemini Embedding 2 (closed) OmniRetriever (ours)Clotho
T → A5.19
19.14 +13.95
Clotho
A → T1.34
16.08 +14.74
SoundDescs
T → A7.00
25.00 +18.00
SoundDescs
A → T7.37
20.70 +13.33Figure 1:Method overview.OmniRetrieveruses the joint embedding zTV A , which is unused by pairwise training
(a), as a supervision target (b) via fusion-as-teacher distillation LDand a Tuple-InfoNCE term LT. This yields
a new open result on 12-direction A VT retrieval (c) and a 13.3 to18.0 R@1 gain over Gemini Embedding 2 on
external audio–text benchmarks (d).
per step. A complementary Tuple-InfoNCE re-
finement (Liu et al., 2021, 2020) LTsupervises
zTV A itself with modality-cycled hard negatives,
preventing the joint vector from collapsing onto the
strongest pair gradient (in practiceT–V).
We instantiate this recipe asOmniRetriever-7B,
an open 7 B A VT retriever. On six standard zero-
shot retrieval benchmarks,OmniRetriever-7Bim-
proves over closed Gemini Embedding 2 by 13to
18R@1 on all four audio–text directions of Clotho
and SoundDescs, reaches the zero-shot CLAP-
family specialist band on Clotho T →A within ∼2
R@1 of SOTA, and matches the contemporary
open zero-shot specialist band on MSR-VTT and
MSVD. To probe the six dual-modal directions
(T↔AV ,A↔TV ,V↔AT ) that no public re-
trieval benchmark currently evaluates, we addition-
ally releaseOmniRetriever-Bench, a 12-direction
A VT retrieval pool of 3,782 held-out triples. On
OmniRetriever-Bench,OmniRetriever-7Breaches
A VG-all 34.84 ,+1.72 over Gemini Embedding 2
and+8.03 over Omni-Embed-Nemotron. A cross-
backbone replication (Section L) reproduces the
dominant LDcontribution at smaller scale, indi-
cating that the recipe is not tied to a particular
backbone.
Contributions.
•Fusion-as-teacher distillation: the joint multi-
modal embedding zTV A of a unified A VT en-
coder is used to supervise its own single-modal
sub-encoders. LDalone gives the dominantsingle-loss gain; the Tuple-InfoNCE term LT
further improves theA↔Vroutes.
•OmniRetriever-7B, an open 7 B A VT retriever. It
improves over closed Gemini Embedding 2 by
13to18R@1 on Clotho and SoundDescs and
reaches the zero-shot audio–text specialist band
on Clotho T →A within ∼2R@1 of SOTA; on
the video–text side it matches the contemporary
open zero-shot specialist band on MSR-VTT and
MSVD.
•OmniRetriever-Bench, a 12-direction A VT re-
trieval pool of 3,782 held-out triples covering
all six single- and six dual-modal directions, the
first public benchmark to evaluate dual-modal
A VT queries.
2 Related Work
Pairwise contrastive vision–language alignment.
CLIP (Radford et al., 2021) and ALIGN (Jia et al.,
2021) establish image–text InfoNCE as the stan-
dard recipe. Subsequent work scales the data
pipeline (Schuhmann et al., 2022; Fang et al., 2024;
Xu et al., 2024), replaces softmax with the sig-
moid loss (Zhai et al., 2023; Tschannen et al.,
2025), and re-anchors with vision foundation mod-
els (Siméoni et al., 2025; Assran et al., 2025; Fini
et al., 2025). A parallel line of work turns MLLMs
into encoders (Wang et al., 2024a; BehnamGhader
et al., 2024; Lee et al., 2025a). Multimodal exten-
sions (VLM2Vec (Jiang et al., 2024), GME (Zhang
et al., 2024), MM-Embed (Lin et al., 2025), Qwen3-
2

INPUTS
 SHARED ENCODER
w e i gh t s  s h a r e d  a c r o s s  a l l  pa t h s
EMBEDDINGS
 OBJECTIVES
anchor
sg
re-use fθhard negativeT
TEXTA person's hand points at the
rear of a red Lotus sports car
parked in a garage…“
V
VIDEO
keyframe
A
AUDIO
o n e  t r i - m o d a l  t r a i n i n g  t u p l e  
  p e r  s t e p(T,V,A)OmniRetriever-7B
Unified  encoder
shared weights BACKBONEfθ
(T,V,A)
θz  TVA
joint anchor
z  T
z  V
z  A
single-modal
z  T~V~A~
negative (cycled HN)Tuple-InfoNCE
Contrast  vs. in-batch tuples + cycled hard negative
(refinement)L  T
z  TVA
z  T~V~A~
Fusion-as-Teacher Distillation
Pull each single-modal embedding  toward the
stop-gradient joint anchor . Tightens the joint
cluster on every modality.PRIMARY
L  D
z  ,z  ,z  TV A
sg(z  ) TVA
Pairwise Alignment
Symmetric pairwise InfoNCE over , ,
.L  A
(z  ,z  ) TV (z  ,z  ) TA
(z  ,z  ) V A
L =λ L + TT λ L + DD λ L AA
MODALITY-CYCLED HARD NEGATIVES  ·  PERIOD-3
1SHUFFLE TEXT
 2SHUFFLE VIDEO
 3SHUFFLE AUDIO
S h u f f l e  e x a c t l y  o n e  m o d a l i t y
wi t h i n  t h e  b a t c h .
re-encoded by
samef  θ
 Cycled hard negative for   —  tightens the joint cluster on e v e r y modality.( , , ) T~V~A~f  ( , , )= θT~V~A~z  T~V~A~
L  TT̃VA T ṼA TVÃ
Figure 2:OmniRetrievertraining overview.A shared encoder fθconsumes the three modalities jointly, producing
the full-modal anchor zTV A, or individually, producing zT,zV,zA.LD(fusion-as-teacher distillation, primary;
Section 3.2) pulls each single-modality embedding toward a stop-gradient copy of zTV A.LT(Tuple-InfoNCE
refinement; Section 3.3) supervises zTV A against the in-batch tuple grid plus a modality-cycled hard negative
z˜T˜V˜A(Equation (4)). LA(pairwise alignment; Section 3.1) ties pairs of single-modality embeddings via symmetric
InfoNCE. At each step the hard negative perturbs one ofT, V, Aon a period-3schedule.
VL-Embedding (Li et al., 2026)) fine-tune MLLMs
with contrastive supervision over LLM-synthesized
triplets. All apply pairwise InfoNCE per modality
combination. Our PAIRWISEbaseline follows this
recipe.
Omni-modal embedding.ImageBind (Girdhar
et al., 2023) and LanguageBind (Zhu et al., 2024)
align N≥3 modalities pairwise to a fixed image or
language anchor. AudioCLIP (Guzhov et al., 2022)
and V ATT (Akbari et al., 2021) optimize three pair-
wise InfoNCE losses over video, audio, and text.
V AST (Chen et al., 2023) adds a 4-way alignment
objective on video–audio–subtitle–caption quadru-
ples. WA VE (Tang et al., 2025) extends Qwen2.5-
Omni with hierarchical visual fusion and a dual
audio encoder but still adds two-modality InfoNCE
losses, as does Omni-Embed-Nemotron (Xu et al.,
2025). In all of these systems, the joint multimodal
embedding is produced as a side effect of pairwise
training and never used as a supervision target in-
side the same backbone.
3 Method
LetD={x i}be an A VT dataset where each xi=
(x(T)
i, x(V)
i, x(A)
i)carries text, video, and audio. Aunified embedder fθ:X →Rdproduces single-
modal embeddings z(m)
i=fθ(x(m)
i)and a joint
embedding z(TV A)
i =fθ(Ti, Vi, Ai)from the same
forward backbone. Figure 2 overviews the three
training objectives.
3.1 Preliminary: Pairwise Alignment (L A)
Existing open A VT embedders (Girdhar et al.,
2023; Zhu et al., 2024; Tang et al., 2025; Xu et al.,
2025) optimize a symmetric InfoNCE loss over
modality pairs:
L(m,m′)
NCE =−1
BX
ilogez(m)⊤
iz(m′)
i/τ
P
jez(m)⊤
iz(m′)
j/τ,(1)
LA=X
(m,m′)∈PL(m,m′)
NCE ,P={(T, V),(T, A),(V, A)}.
The PAIRWISEbaseline we report throughout
usesLAalone. Since LAnever operates on the
joint(T, V, A) vector, zTV A is neither supervised
nor distilled: pairwise alignment binds the three
streams only to the extent that each pair enforces in
isolation.OmniRetrieverretains LAand adds two
joint-level losses,L DandL T, described next.
3.2 Fusion-as-Teacher Distillation (L D)
The joint forward of fθon(T, V, A) is the only step
at which all three modalities interact in the network.
3

We use zTV A as the teacher. A symmetric InfoNCE
pulls each single-modal embedding toward a stop-
gradient copy of zTV A and pushes it away from the
joint vectors of other in-batch samples:
LD=1
|M|X
m∈ML(m,sg(TV A))
NCE,(2)
withM={T, V, A} . The teacher sg(z TV A)al-
ready encodes the three modalities jointly, so each
single-modal student is trained against the joint ge-
ometry that an A+T+V query reaches at inference.
Teacher and student share fθand live in the same
batch; the only additional cost over the pairwise
baseline is one extra joint forward pass.
Fusion vs. unimodal teacher.A unimodal
teacher gm(x(m)), e.g. SigLIP (Tschannen et al.,
2025), Whisper, or BEATs (Chen et al., 2022),
is trained on the marginal p(x(m))and there-
fore carries no information about cross-modal
neighbors of x. By contrast, zTV A is computed
on the joint input, so distillation into zApropa-
gates ∂sim(z A,sg(z TV A))/∂θ A, which encodes
theT, V neighbors that an audio-only query will
see at inference. This predicts two effects, both
of which we confirm. Audio-anchored directions
benefit most from LD(audio-related mean +3.73
vs. video-only+2.48, Table S11).
3.3 Tuple-InfoNCE Refinement (L T)
LTis a regularizer that keeps zTV A informative
about all three modalities, preventing it from col-
lapsing onto a dominant pair. The joint vector
zTV A thatLDuses as a teacher remains a pas-
sive average of three pair geometries on top of pair-
wise alignment: LAnever back-propagates through
zTV A, so matched triples are not pulled tighter
than mismatched ones, and the contribution of each
modality is determined by whichever pair gradient
is strongest (in practice T–V). Audio is under-
represented in zTV A, a known modality-imbalance
effect in multimodal fusion (Wang et al., 2020;
Peng et al., 2022).L Tsupervisesz TV A directly.
Tuple-InfoNCE.For a batch of size Band
modality set M={T, V, A} , the joint similarity of
an index pair (i, j) averages the six cross-modal
cosines,
s(i, j)=1
M(M−1)X
m̸=m′z(m)⊤
iz(m′)
j,(3)and the Tuple-InfoNCE loss (Liu et al., 2021, 2020)
reads
LT=−1
BX
iloges(i,i)/τTP
jes(i,j)/τT+es(i,˜i)/τT,(4)
where ˜iis a modality-cycled hard negative de-
scribed below. When M≥3 ,s(i, j) is minimized
over mismatch inanycross-modal direction, so LT
assigns the largest gradient to the modality direc-
tion along which the matched triple is currently
slackest, providing the joint-level supervision miss-
ing from pairwise alignment.
Modality-cycled hard negatives.The in-batch
grid in Equation (4) contributes B−1 negatives, but
each differs from the anchor inall threemodalities,
so its gradient pushes the joint cluster apart glob-
ally without specifying which modality direction
is too slack. We therefore construct one targeted
negative per anchor by shufflingexactly onemodal-
ity of the batch with a derangement, producing
a tuple that disagrees with the anchor in a single
slot; the resulting contrastive gradient tightens the
joint cluster along that one direction. To prevent
the joint geometry from drifting back to a T+V -
dominated configuration once any single modality
is tightened, we cycle the shuffled slot determinis-
tically across {T, V, A} with period 3, supervising
every modality direction in turn.
The three shuffled slots contribute asymmetri-
cally. The audio-shuffle slot produces the con-
trastive gradient least redundant with the in-batch
grid (audio carries the largest caption gap in our
training data). The text- and video-shuffle slots
prevent the joint cluster from drifting back to a
T+V -dominated geometry. We use k= 1 per an-
chor; higher koverfits to modality-imbalanced mis-
matches (Thakur et al., 2025). LDprovides the
dominant single-loss gain, and LTimproves the
A↔Vroutes on top of it (Table S11).
3.4 Final Objective
OmniRetrieveroptimizes
LOmniRetriever =λDLD+λTLT+λALA,(5)
with(λD, λT, λA)=(1,1,1) chosena priori(not
tuned onOmniRetriever-Bench); Section D shows
that the recipe is insensitive to the loss-weight ra-
tio across a wide range. Algorithm 1 gives the
per-step training loop; the per-step cost over the
pairwise baseline is one extra joint forward pass,
and inference latency is reported in Section C.
4

Algorithm 1OmniRetrievertraining step (batch B,
stept).
1:Forward: z(T),z(V),z(A)←f θ(3 single-modal);
z(TV A)←fθ(T, V, A)(1 joint)
2:PairwiseL A←P
(m,m′)∈{(T,V),(T,A),(V,A)}L(m,m′)
NCE
3:DistillL D←1
3P
m∈{T,V,A}L(m,sg(z(TV A)))
NCE
4:Cycled HN: mt← {T, V, A}[tmod 3] ; draw derange-
mentσ; build ˜i=(. . . , x(mt)
σ(i), . . .)
5:TupleL T←−1
BP
iloges(i,i)/τT
P
jes(i,j)/τT+es(i,˜i)/τT
6:Updateθ←θ−η∇ θ(λDLD+λTLT+λALA)
4 Experiments
4.1 Setup
OmniRetriever-7Bis an adapter fine-tune of the
open-weights WA VE-7B (Tang et al., 2025) back-
bone, with LoRA adapters on the LLM trunk, an
all-layer fusion head, and a BEATs adaptor; full
architectural details are in Section A. The train-
ing data is a 1.5M-triple subset sampled from four
public video–text datasets, InternVid (Wang et al.,
2024b), InternVid-FLT (Wang et al., 2024b), Panda-
70M (Chen et al., 2024), and PVD (Bolya et al.,
2026), together with a small in-house video cor-
pus collected with consent. We restrict the pool to
clips that contain text, video, and audio (Section E).
We release model weights, evaluation code, and
OmniRetriever-Bench. The training set is sample-
identifier disjoint from every evaluation pool in
this paper, and the third-party benchmarks (Clotho,
SoundDescs, MSR-VTT, MSVD, DiDeMo, V A-
TEX) are curated by other groups under different
caption styles.
We compareOmniRetriever-7B(ours) against
three external systems: the frozen open-weights
backbone WA VE-7B (Tang et al., 2025), the open
Omni-Embed-Nemotron (Xu et al., 2025), and
closed Gemini Embedding 2 via Google’s offi-
cial endpoint with default settings. The endpoint
ingests raw audio (WA V) and raw video (MP4
with audio muxed) as inline data, so Gemini re-
ceives the same clip bytes asOmniRetriever-7B.
We cannot, however, inspect how the closed sys-
tem internally routes inline audio, so the Gemini
audio numbers reflect the currently deployed multi-
part product rather than a model-capacity ceiling.
Reported numbers are the mean over three seeds
{42,43,44} ; aggregate seed std is ≤0.18 R@1,
which we treat as the noise floor. The PAIRWISE
ablation ( LAalone, the recipe used by recent uni-
fied embedders (Wei et al., 2024; Jiang et al., 2024;Zhang et al., 2024)) is reported in Table 5.
Systems we do not compare against.Qwen3-
VL-Embedding (Li et al., 2026) has no audio path
and cannot serve the ten audio-anchored directions.
A head-to-head comparison is left to future work.
The CLAP audio–text specialists (Table 2) have no
video stream and cannot serve the sixV-anchored
directions.
4.2 Results on Standard Audio–Text
Benchmarks
We evaluateOmniRetriever-7Bzero-shot on
two standard audio–text retrieval benchmarks,
Clotho (Drossos et al., 2020) and SoundDe-
scs (Koepke et al., 2022), and compare with the
published audio–text specialist literature, contem-
porary open omni-modality embedders, and closed
Gemini Embedding 2. Table 2 reports R@1; full
R@kand NDCG@10 are in Table S10.
OmniRetriever-7Bimproves over closed Gemini
Embedding 2 by 13.9 to18.0 R@1 on T →A and
13.3 to14.7 on A→T. Gemini’s audio-anchored
R@1 is consistently low across benchmarks ( 1.34
on Clotho A →T,1.48 onOmniRetriever-Bench
A→T below), suggesting that its deployed multi-
part endpoint does not route audio competitively for
retrieval. The CLAP-family specialists are audio–
text dual-tower models trained on AudioCaps-
style pairs without video exposure, and form the
strongest single-task audio-retrieval band ( 19to21
R@1 on Clotho T →A).OmniRetriever-7Breaches
19.14 on Clotho T →A, within ∼2R@1 of this
band. On Clotho A →T,OmniRetriever-7Breaches
16.08 , still below the specialist band ( 25to26.5)
but well above prior open omni-modality sys-
tems on the same direction (ImageBind reports
6.0T→A, LanguageBind 16.7 T→A; neither re-
ports A →T). On SoundDescs T →A and A →T,
OmniRetriever-7Breaches 25.00 and20.70 , the
strongest open-system numbers in our literature
search. Scaling the training corpus is expected to
pushOmniRetriever-7B’s A →T performance to-
ward and potentially beyond the specialist band,
since cross-modal supervision scales naturally with
additional multimodal data.
4.3 Results on Standard Video–Text
Benchmarks
We evaluateOmniRetriever-7Bzero-shot on four
standard video–text benchmarks (MSR-VTT (Xu
et al., 2016), MSVD (Chen and Dolan, 2011),
5

Table 1:Comparison with prior cross-modal retrieval benchmarks.Each column underSingle-modalor
Dual-modalmarks whether the benchmark supports a given retrieval direction-pair (a ✓indicates both directions of
the pair are evaluated).Totalis the resulting number of retrieval directions and Nis the size of the standard test
pool.OmniRetriever-Benchis the first public benchmark to cover all three A VT modalities in one pool and the first
to evaluate dual-modal queries.
Single-modal pairs Dual-modal pairs
BenchmarkT↔V T↔A V↔A T↔AV A↔TV V↔ATTotalN
Video–text retrieval
MSR-VTT (Xu et al., 2016)✓– – – – – 21,000
MSVD (Chen and Dolan, 2011)✓– – – – – 2670
DiDeMo (Anne Hendricks et al., 2017)✓– – – – – 21,004
V ATEX (Wang et al., 2019)✓– – – – – 24,468
Audio–text retrieval
Clotho (Drossos et al., 2020) –✓– – – – 21,045
SoundDescs (Koepke et al., 2022) –✓– – – – 21,000
AudioCaps (Kim et al., 2019) –✓– – – – 2975
OmniRetriever-Bench(ours) ✓ ✓ ✓ ✓ ✓ ✓ 12 3,782
DiDeMo (Anne Hendricks et al., 2017), V A-
TEX (Wang et al., 2019)) and compare with pub-
lished specialist literature and contemporary omni-
modality embedders. Table 3 reports R@1; full
R@kand NDCG@10 are in Table S9.
OmniRetriever-7Breaches 47.6/43.7 on
MSR-VTT, matching PE-coreB (Bolya et al.,
2026) ( 47.6/47.3 ) and improving over SigLIP-2-
L (Tschannen et al., 2025). On MSVD it reaches
66.9/63.3 , improving over PE-coreL ( 57.2 T→V).
On DiDeMo and V ATEX,OmniRetriever-7B
trails the strongest specialist with results on
both benchmarks, InternVideo2-1B (Wang et al.,
2024c), by 9to12R@1 on three of the four
directions and by 30R@1 on V ATEX V →T,
driven by InternVideo2’s exceptional 85.4 V→T.
We attribute this gap to the data scale of single-task
video training;OmniRetriever-7Balso has no audio
specialisation for these benchmarks, which carry
no audio queries.OmniRetriever-7Bimproves over
LanguageBind on 5of6reported video directions.
Closed Gemini wins all eight video directions by 5
to15R@1, consistent with reports that Gemini is
trained on a substantially larger closed video–text
corpus; a data-parity comparison is out of scope.
Taken together with the audio results in Table 2,
OmniRetriever-7Bis the strongest open unified
omni-modal embedder in our literature search.
Why a unified A VT embedder.Pairing a CLAP-
style audio specialist with an InternVideo2 video
specialist covers four of the twelve directions, but
neither system produces embeddings for the six
dual-modal directions ( T↔AV , etc.) in oneshared index, and neither can anchor an audio query
against text grounded in shared video context.Om-
niRetrievertargets the setting in which a single
shared(T, V, A)index is required.
OmniRetriever-7Breduces the audio-capability
gap among general-purpose omni-modality embed-
ders, reaches the zero-shot audio–text specialist
band on Clotho T →A, and stays within reach of
single-task video–text specialists while covering
all twelve directions in one model.
4.4OmniRetriever-Bench: A 12-Direction
A VT Probe
The benchmarks above evaluate single-modal di-
rections only and do not stress-test the joint repre-
sentation zTV A thatOmniRetrieveris built around.
A practical A VT retriever also needs to serve six
dual-modal queries ( T↔AV ,A↔TV ,V↔
AT) that compose two modalities on one side.
No public suite covers these directions: standard
video–text and audio–text benchmarks each evalu-
ate two single-modal directions, never both video
and audio in the same pool. Table 1 contrasts
OmniRetriever-Benchwith these prior benchmarks
across modality coverage, number of retrieval di-
rections, and evaluation-pool size.
OmniRetriever-Benchcontains 3,782 held-out
aligned (T, V, A) triples, scored against the same
gallery in every direction; the metric is Recall@1.
Triples are curated from in-house video sources that
are disjoint from the training data at the sample-
identifier level.All 3,782 captions are reviewed
and corrected by trained human annotators:
each caption starts from a Gemini 3.0 Pro draft,
6

Table 2:Audio–text retrieval R@1 on Clotho and SoundDescs (zero-shot).Specialised retrievers fine-tuned on
each task are marked FT. Cited rows draw on the original papers’ published values.OmniRetriever-7Boutperforms
closed Gemini Embedding 2 by 13.3 to18.0 R@1 in every direction; on Clotho T →AOmniRetriever-7B’s 19.14
sits within ∼2R@1 of the current zero-shot contrastive SOTA band (Cacophony, MGA-CLAP, M2D-CLAP at 20
to21) without task-specific fine-tuning.
Clotho SoundDescs
Model Year Setting T→A A→T T→A A→T
Audio–text specialists
MMT (Koepke et al., 2022) 2022 FT 6.7 7.0 30.7 31.4
MS-CLAP 2023 ZS 16.7 20.0 — —
LAION-CLAP fusion (Wu et al., 2023) 2023 ZS 18.2 25.7 — —
WavCaps-HTSAT (Mei et al., 2024) 2023 ZS 19.5 23.4 — —
WavCaps-CNN14 (Mei et al., 2024) 2023 ZS21.225.9 — —
FLAP fusion 2024 ZS 20.3 25.5 — —
Cacophony 2024 ZS 20.2 26.5 — —
MGA-CLAP 2024 ZS 20.826.5— —
GLAP 2025 ZS 19.4 21.8 — —
M2D-CLAP (Niizumi et al., 2025) 2025 ZS 20.1 25.0 — —
Closed-source industrial baselines
Gemini Embedding 2 2026 ZS 5.2 1.3 7.0 7.4
Open-source omni-modality embedders
ImageBind (ViT-H) (Girdhar et al., 2023) 2023 ZS 6.0 — — —
LanguageBind (CLIP-H/14) (Zhu et al., 2024) 2024 ZS 16.7 — — —
Omni-Embed-Nemotron (Xu et al., 2025) 2025 ZS 6.4 3.5 6.4 4.8
OmniRetriever-7B(ours) 2026 ZS 19.1 16.1 25.0 20.7
then the annotator verifies alignment against the
video and the audio track and rewrites any inaccura-
cies. Research-use licensing follows prior released
benchmarks built on proprietary content (Wang
et al., 2019; Faysse et al., 2025); full annotation
guidelines, the distribution mismatch with the train-
ing corpus, and a caption-paraphrase robustness
check are in Sections E, F and M.
Table 4 reports R@1 on all 12 directions of
OmniRetriever-Bench.OmniRetriever-7Breaches
A VG-all 34.84 ,+9.52 over the frozen WA VE-7B
backbone, +8.03 over Omni-Embed-Nemotron,
and+1.72 over closed Gemini Embedding 2
(33.12 ). Both gaps over external systems exceed
the seed noise floor ( ≤0.18 R@1 std). The gains
concentrate on the eight audio-anchored directions:
A→T ,11.92 vs.1.48 (+10.44 );A→T+V ,
23.45 vs.6.00;V→A ,25.46 vs.13.80 . Gem-
ini wins the four visually-saturated T↔V and
T↔A+V routes by 6to10R@1, consistent with
its larger closed video–text corpus. A qualitative
comparison is in Section O.
4.5 Ablation and Analysis
Table 5 reports the stair-step ablation against the
frozen WA VE-7B base onOmniRetriever-Bench,
isolating the contribution of each loss in theOm-
niRetrieverobjective. The per-direction breakdown
is in Table S11.
Pairwise baseline.The PAIRWISEbaseline ( LA
only, on our training corpus) lifts A VG-all from25.32 to31.08 (+5.76 ). The gain is unevenly dis-
tributed: T↔V and most dual-modal directions
improve by 5to9R@1, but audio-anchored R@1
stays low ( a→t9.00 ,a→tv17.00 ). The audio
axis is not capacity-bottlenecked here; it is bot-
tlenecked by the weak-fusion regime of pairwise
alignment.
Fusion-as-teacher distillation ( LD).Adding
LDlifts A VG-all by +3.52 (31.08→34.60 ), the
dominant single-loss contribution. This already im-
proves on closed Gemini Embedding 2 by +1.48
R@1. The gain spans both regimes (video-only
+2.48 , audio-related +3.73 ; Table S11); the audio
side gains more because the joint forward routes
T, V context into the audio sub-encoder, which
unimodal teachers cannot do.
Tuple-InfoNCE refinement ( LT).Adding LT
on top of LD+LAlifts A VG-all by +0.24 to34.84 .
The aggregate gain is small but stable across seeds:
the per-seed ∆LTis{+0.24,+0.21,+0.27} on
{42,43,44} (Table S15), well above the aggregate
seed std ( ≤0.07 ). The per-direction breakdown
(Table S11) shows that LTredistributes capacity:
it adds +2.03 to+2.86 R@1 on the four A↔V
routes (above per-direction noise of ±0.5 to±0.8 )
while losing ∼2R@1 on text-anchored dual-modal
routes. We keep LTin the released model because
A↔V is the bottleneck direction for open A VT
embedders.
7

Table 3:Video–text retrieval R@1 on four standard benchmarks (zero-shot).Specialised video–text retrievers
fine-tuned per task are marked FT; the remainder are zero-shot. Cited rows draw on each paper’s published values.
OmniRetriever-7Bmatches the contemporary zero-shot specialist band on MSR-VTT and MSVD and trails the
strongest specialist with DiDeMo/V ATEX results, InternVideo2-1B, by ∼9to∼12 R@1 on three of those four
directions (and by∼30R@1 on the exceptional V ATEX V→T). “—” denotes a number not reported.
T→V(or T→A V)V→T(or A V→T)
Model Setting MSR-VTT MSVD DiDeMo V ATEX MSR-VTT MSVD DiDeMo V ATEX
Specialised video–text retrievers (FT)
CLIP4Clip (Luo et al., 2021) FT 44.5 46.2 43.4 55.9 42.7 56.0 42.5 73.4
X-CLIP FT 49.3 50.4 47.8 — 48.9 66.8 47.8 —
Zero-shot video–text encoder band (specialists)
SigLIP-2 family (Tschannen et al., 2025) ZS 38.5–43.1 49.0–55.8 — — 30.1–34.3 67.2–74.6 — —
PE-core family (Bolya et al., 2026) ZS 47.6–51.2 50.4–59.7 — — 47.3–50.1 76.7–85.4— —
InternVideo2-1B (Wang et al., 2024c) ZS 51.9 58.157.070.4 50.983.354.3 85.4
Closed-source industrial baselines
Gemini Embedding 2 ZS53.9 77.1 55.6 69.4 48.3 77.9 53.3 66.7
Open-source omni-modality embedders
ImageBind (ViT-H) (Girdhar et al., 2023) ZS 36.1 — — — — — — —
LanguageBind (CLIP-H/14) (Zhu et al., 2024) ZS 44.8 53.9 39.9 — 40.9 72.0 39.8 —
Omni-Embed-Nemotron (Xu et al., 2025) ZS 35.8 55.8 41.9 47.5 30.6 49.2 36.5 39.9
OmniRetriever-7B(ours) ZS 47.6 66.9 46.0 58.0 43.7 63.3 45.1 55.0
Table 4:Recall@1 on the 12 directions ofOmniRetriever-Bench. Bold: best per direction. Theoursrow is
shaded. The pairwise-only baseline (PAIRWISE,L Aalone) is reported in Table 5.
Single-modal Dual-modal Averages
Systemt→v v→t t→a a→t v→a a→v t→av av→t a→tv tv→a v→at at→vsingle dual all
Closed-source industrial baselines
Gemini Embedding 255.13 53.8312.61 1.48 13.80 15.7655.45 50.166.00 16.7954.97 61.4525.44 40.80 33.12
Open-source omni-modality embedders
WA VE-7B (Tang et al., 2025) 41.27 34.45 8.67 3.38 14.89 12.93 44.95 42.68 2.27 15.65 38.74 43.92 19.27 31.37 25.32
Omni-Embed-Nemotron (Xu et al., 2025) 43.55 39.74 12.19 8.83 14.07 12.35 36.12 34.53 14.75 16.63 41.30 47.67 21.79 31.84 26.81
OmniRetriever (ours) 48.89 46.30 14.20 11.92 25.46 24.99 45.37 44.47 23.45 26.07 52.49 54.47 28.63 41.05 34.84
Table 5:Ablation of theOmniRetrieverlosses on
OmniRetriever-Bench(A VG-all R@1; ∆vs. the frozen
WA VE-7B base). LDon top of the PAIRWISEbaseline
adds +3.52 R@1, already +1.48 over closed Gemini
Embedding 2 ( 33.12 ).LTadds a further +0.24 on
aggregate and redistributes capacity toward A↔V (Ta-
ble S11).
Variant A VG all∆
WA VE-7B (no fine-tune) 25.32 —
PAIRWISE(L Aonly) 31.08+5.76
LD+LA(noL T) 34.60+9.28
OmniRetriever-7B(L T+LD+LA)34.84 +9.52
Discussion.Pairwise InfoNCE optimizes three
pair softmaxes independently, so the joint (T, V, A)
vector inherits whichever pair carries the largest co-
sine gap (in practice T–V) and leaves the audio
axis slack. LDsupplies the cross-modal teacher
that closes the audio gap at the single-modal level.
LTthen redistributes the remaining joint-level ca-
pacity toward the A↔V axis. We conjecture
that Gemini Embedding 2 follows a similar pair-
wise contrastive recipe based on public reports(Lee
et al., 2025b); our PAIRWISEbaseline performs
similarly to it in the same weak-fusion regime.5 Conclusion
A unified A VT encoder can computes a joint em-
bedding zTV A on every (T, V, A) forward, but pair-
wise InfoNCE leaves it unsupervised. We turn
zTV A into a training signal: LDdistills its stop-
gradient copy into the single-modal embeddings,
andLTsupervises it directly against a tuple grid
with modality-cycled hard negatives. Both losses
reuse the same backbone forward and apply to any
unified retriever whose forward produces a joint
embedding. A cross-backbone replication repro-
duces the dominant LDeffect.OmniRetriever-7B
reaches the zero-shot audio–text specialist band
on Clotho T →A, reduces the open omni-modality
gap on A →T, and improves over closed Gemini
Embedding 2 onOmniRetriever-Benchby 1.72
A VG-all R@1. To probe the joint representation
that pairwise training cannot expose, we release
OmniRetriever-Bench, the first A VT benchmark
scoring 3,782 human-corrected triples on a shared
gallery across all 12 single- and dual-modal di-
rections. Model weights, training code, and the
benchmark are released.
8

Limitations
Video–text gap.On the four T↔V and
T↔A+V directions,OmniRetriever-7Btrails
closed Gemini Embedding 2 by 6to10R@1 on
OmniRetriever-Benchand by 5to15R@1 on the
external video benchmarks. We attribute this gap to
data scale: Gemini is trained on closed video–text
corpora reportedly orders of magnitude larger than
what is practically reachable in an academic A VT
setting, and a unified (T, V, A) embedder at our
training scale is not expected to match dedicated
video–text encoders on every benchmark.
Closed-baseline comparability.Gemini Embed-
ding 2 is accessed only through a deployed mul-
tipart API. We feed it the same raw audio+video
bytesOmniRetriever-7Bconsumes, but we cannot
inspect how the closed system internally routes in-
line audio. The Gemini numbers therefore reflect
the deployed product rather than a model-capacity
ceiling.
Backbone coverage.OmniRetrieveris instanti-
ated on a 7 B WA VE backbone, with a 3B Omni-
Embed-Nemotron replication in Section L. Ver-
ification on other backbones (e.g., Qwen3-VL-
Embedding) is left to future work.
Compression.Our analyses focus on retrieval
accuracy. We do not study embedding compression
beyond the post-hoc int8/binary baselines reported
in Section G; a compression-aware training recipe
is left to future work.
Ethical Considerations
Privacy.Embedding inversion (vec2text (Morris
et al., 2023)) is a known attack on retrieval embed-
dings. The training corpus is sampled from four
public video–text datasets (InternVid, InternVid-
FLT, Panda-70M, PVD) and a small in-house cor-
pus collected with consent; we use the public
sources under the terms of their original releases
and do not redistribute the assembled training sub-
set. We recommend application-layer defences
such as DP-SGD or representation distortion for de-
ployments with sensitive content.OmniRetriever-
Benchreleases only the source identifiers and cap-
tion text needed to reproduce retrieval scores.
Intended use.We releaseOmniRetriever-7Band
OmniRetriever-Benchfor research on cross-modal
retrieval. The research-use license accompanyingthe release prohibits deployment in surveillance
applications affecting natural persons.
Provenance and licensing.The training corpus
combines four public video–text datasets (Intern-
Vid, InternVid-FLT, Panda-70M, PVD) with a
small in-house corpus collected with consent. We
do not redistribute the assembled training subset.
OmniRetriever-Benchis built from sources held
out from training and is released following the stan-
dard video-benchmark protocol: source identifiers,
clip intervals, and captions are released, while the
underlying media is not redistributed. Curation
details are in Section F.
References
Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong
Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong.
2021. Vatt: Transformers for multimodal self-
supervised learning from raw video, audio and text.
Advances in neural information processing systems,
34:24206–24221.
Lisa Anne Hendricks, Oliver Wang, Eli Shechtman,
Josef Sivic, Trevor Darrell, and Bryan Russell. 2017.
Localizing moments in video with natural language.
InProceedings of the IEEE international conference
on computer vision, pages 5803–5812.
Mido Assran, Adrien Bardes, David Fan, Quentin Gar-
rido, Russell Howes, Matthew Muckley, Ammar
Rizvi, Claire Roberts, Koustuv Sinha, Artem Zho-
lus, and 1 others. 2025. V-jepa 2: Self-supervised
video models enable understanding, prediction and
planning.arXiv preprint arXiv:2506.09985.
Parishad BehnamGhader, Vaibhav Adlakha, Marius
Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and
Siva Reddy. 2024. Llm2vec: Large language models
are secretly powerful text encoders.arXiv preprint
arXiv:2404.05961.
Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun
Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale
Zhi, Jathushan Rajasegaran, Hanoona Bangalath, and
1 others. 2026. Perception encoder: The best visual
embeddings are not at the output of the network.Ad-
vances in Neural Information Processing Systems,
38:60884–60937.
David Chen and William B Dolan. 2011. Collecting
highly parallel data for paraphrase evaluation. In
Proceedings of the 49th annual meeting of the associ-
ation for computational linguistics: human language
technologies, pages 190–200.
Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu,
Daniel Tompkins, Zhuo Chen, and Furu Wei. 2022.
Beats: Audio pre-training with acoustic tokenizers.
arXiv preprint arXiv:2212.09058.
9

Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao,
Mingzhen Sun, Xinxin Zhu, and Jing Liu. 2023. Vast:
A vision-audio-subtitle-text omni-modality founda-
tion model and dataset.Advances in Neural Informa-
tion Processing Systems, 36:72842–72866.
Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace,
Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun
Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-
Hsuan Yang, and 1 others. 2024. Panda-70m: Cap-
tioning 70m videos with multiple cross-modality
teachers. InProceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition.
Konstantinos Drossos, Samuel Lipping, and Tuomas
Virtanen. 2020. Clotho: An audio captioning dataset.
InICASSP 2020-2020 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing
(ICASSP), pages 736–740. IEEE.
Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig
Schmidt, Alexander Toshev, and Vaishaal Shankar.
2024. Data filtering networks. InInternational Con-
ference on Learning Representations, volume 2024,
pages 36221–36237.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. Colpali: Efficient document retrieval with
vision language models. InInternational Conference
on Learning Representations, volume 2025, pages
61424–61449.
Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter,
Michal Klein, David Haldimann, Sai Aitharaju, Vic-
tor G Turrisi da Costa, Louis Béthune, Zhe Gan,
and 1 others. 2025. Multimodal autoregressive pre-
training of large vision encoders. InProceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 9641–9654.
Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Man-
nat Singh, Kalyan Vasudev Alwala, Armand Joulin,
and Ishan Misra. 2023. Imagebind: One embed-
ding space to bind them all. InProceedings of the
IEEE/CVF conference on computer vision and pat-
tern recognition, pages 15180–15190.
Andrey Guzhov, Federico Raue, Jörn Hees, and An-
dreas Dengel. 2022. Audioclip: Extending clip to
image, text and audio. InICASSP 2022-2022 IEEE
International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 976–980. IEEE.
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana
Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen
Li, and Tom Duerig. 2021. Scaling up visual and
vision-language representation learning with noisy
text supervision. InInternational conference on ma-
chine learning, pages 4904–4916. PMLR.
Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz,
Yingbo Zhou, and Wenhu Chen. 2024. Vlm2vec:
Training vision-language models for massive
multimodal embedding tasks.arXiv preprint
arXiv:2410.05160.Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee,
and Gunhee Kim. 2019. Audiocaps: Generating cap-
tions for audios in the wild. InProceedings of the
2019 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 119–132.
A Sophia Koepke, Andreea-Maria Oncescu, João F Hen-
riques, Zeynep Akata, and Samuel Albanie. 2022.
Audio retrieval with natural language queries: A
benchmark study.IEEE Transactions on Multimedia,
25:2675–2685.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2025a. Nv-embed: Improved techniques
for training llms as generalist embedding models. In
International Conference on Learning Representa-
tions, volume 2025, pages 79310–79333.
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel
Cer, Madhuri Shanbhogue, Iftekhar Naim, Gus-
tavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Hen-
rique Schechter Vera, and 1 others. 2025b. Gemini
embedding: Generalizable embeddings from gemini.
arXiv preprint arXiv:2503.07891.
Mingxin Li, Yanzhao Zhang, Dingkun Long, Ke-
qin Chen, Sibo Song, Shuai Bai, Zhibo Yang,
Pengjun Xie, An Yang, Dayiheng Liu, and 1 oth-
ers. 2026. Qwen3-vl-embedding and qwen3-vl-
reranker: A unified framework for state-of-the-art
multimodal retrieval and ranking.arXiv preprint
arXiv:2601.04720.
Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi,
Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2025.
Mm-embed: Universal multimodal retrieval with
multimodal llms. InInternational Conference
on Learning Representations, volume 2025, pages
44215–44234.
Yunze Liu, Qingnan Fan, Shanghang Zhang, Hao Dong,
Thomas Funkhouser, and Li Yi. 2021. Contrastive
multimodal fusion with tupleinfonce. InProceed-
ings of the IEEE/CVF international conference on
computer vision, pages 754–763.
Yunze Liu, Li Yi, Shanghang Zhang, Qingnan
Fan, Thomas Funkhouser, and Hao Dong. 2020.
P4contrast: Contrastive learning with pairs of point-
pixel pairs for rgb-d scene understanding.arXiv
preprint arXiv:2012.13089.
Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen
Lei, Nan Duan, and Tianrui Li. 2021. Clip4clip:
An empirical study of clip for end to end video clip
retrieval.arXiv preprint arXiv:2104.08860.
Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan,
Ji Zhang, and Rongrong Ji. 2022. X-clip: End-to-end
multi-grained contrastive learning for video-text re-
trieval. InProceedings of the 30th ACM international
conference on multimedia, pages 638–647.
10

Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang
Kong, Tom Ko, Chengqi Zhao, Mark D Plumbley,
Yuexian Zou, and Wenwu Wang. 2024. Wavcaps:
A chatgpt-assisted weakly-labelled audio caption-
ing dataset for audio-language multimodal research.
IEEE/ACM Transactions on Audio, Speech, and Lan-
guage Processing, 32:3339–3354.
John Morris, V olodymyr Kuleshov, Vitaly Shmatikov,
and Alexander M Rush. 2023. Text embeddings
reveal (almost) as much as text. InProceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing, pages 12448–12460.
Daisuke Niizumi, Daiki Takeuchi, Masahiro Yasuda,
Binh Thien Nguyen, Yasunori Ohishi, and Noboru
Harada. 2025. M2d-clap: Exploring general-purpose
audio-language representations beyond clap.IEEE
Access.
Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang,
and Di Hu. 2022. Balanced multimodal learning via
on-the-fly gradient modulation. InProceedings of
the IEEE/CVF conference on computer vision and
pattern recognition, pages 8238–8247.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark, and
1 others. 2021. Learning transferable visual models
from natural language supervision. InInternational
conference on machine learning, pages 8748–8763.
PmLR.
Christoph Schuhmann, Romain Beaumont, Richard
Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti,
Theo Coombes, Aarush Katta, Clayton Mullis,
Mitchell Wortsman, and 1 others. 2022. Laion-5b:
An open large-scale dataset for training next genera-
tion image-text models.Advances in neural informa-
tion processing systems, 35:25278–25294.
Oriane Siméoni, Huy V V o, Maximilian Seitzer, Fed-
erico Baldassarre, Maxime Oquab, Cijo Jose, Vasil
Khalidov, Marc Szafraniec, Seungeun Yi, Michaël
Ramamonjisoa, and 1 others. 2025. Dinov3.arXiv
preprint arXiv:2508.10104.
Changli Tang, Qinfan Xiao, Ke Mei, Tianyi Wang,
Fengyun Rao, and Chao Zhang. 2025. Wave: Learn-
ing unified & versatile audio-visual embeddings with
multimodal llm.arXiv preprint arXiv:2509.21990.
Nandan Thakur, Crystina Zhang, Xueguang Ma, and
Jimmy Lin. 2025. Hard negatives, hard lessons: Re-
visiting training data quality for robust information
retrieval with llms.arXiv preprint arXiv:2505.16967.
Michael Tschannen, Alexey Gritsenko, Xiao Wang,
Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin,
Nikhil Parthasarathy, Talfan Evans, Lucas Beyer,
Ye Xia, Basil Mustafa, and 1 others. 2025. Siglip
2: Multilingual vision-language encoders with im-
proved semantic understanding, localization, and
dense features.arXiv preprint arXiv:2502.14786.Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024a. Improv-
ing text embeddings with large language models. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 11897–11916.
Weiyao Wang, Du Tran, and Matt Feiszli. 2020. What
makes training multi-modal classification networks
hard? InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages
12695–12705.
Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-
Fang Wang, and William Yang Wang. 2019. Vatex:
A large-scale, high-quality multilingual dataset for
video-and-language research. InProceedings of the
IEEE/CVF international conference on computer vi-
sion, pages 4581–4591.
Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo
Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen,
Yaohui Wang, and 1 others. 2024b. Internvid: A
large-scale video-text dataset for multimodal under-
standing and generation. InThe Twelfth International
Conference on Learning Representations.
Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yi-
nan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun
Wang, Yansong Shi, and 1 others. 2024c. Intern-
video2: Scaling foundation models for multimodal
video understanding. InEuropean conference on
computer vision, pages 396–416. Springer.
Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu,
Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen.
2024. Uniir: Training and benchmarking univer-
sal multimodal information retrievers. InEuropean
Conference on Computer Vision, pages 387–404.
Springer.
Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Tay-
lor Berg-Kirkpatrick, and Shlomo Dubnov. 2023.
Large-scale contrastive language-audio pretraining
with feature fusion and keyword-to-caption augmen-
tation. InICASSP 2023-2023 IEEE International
Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 1–5. IEEE.
Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang,
Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi
Ghosh, Luke Zettlemoyer, and Christoph Feichten-
hofer. 2024. Demystifying clip data. InInternational
Conference on Learning Representations, volume
2024, pages 47812–47831.
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. Msr-
vtt: A large video description dataset for bridging
video and language. InProceedings of the IEEE con-
ference on computer vision and pattern recognition,
pages 5288–5296.
Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel
Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even
Oldridge, and Benedikt Schifferer. 2025. Omni-
embed-nemotron: A unified multimodal retrieval
11

model for text, image, audio, and video.arXiv
preprint arXiv:2510.03458.
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov,
and Lucas Beyer. 2023. Sigmoid loss for language
image pre-training. InProceedings of the IEEE/CVF
international conference on computer vision, pages
11975–11986.
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi
Dai, Dingkun Long, Pengjun Xie, Meishan Zhang,
Wenjie Li, and Min Zhang. 2024. Gme: improving
universal multimodal retrieval by multimodal llms.
arXiv preprint arXiv:2412.16855.
Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui,
WANG HongFa, Yatian Pang, Wenhao Jiang, Junwu
Zhang, Zongwei Li, and 1 others. 2024. Language-
bind: Extending video-language pretraining to n-
modality by language-based semantic alignment. In
International Conference on Learning Representa-
tions, volume 2024, pages 9588–9608.
Appendix Overview
•Sections A and B: backbone, LoRA configura-
tion, and full training hyper-parameters.
•Section C: end-to-end inference latency on the
four input paths.
•Section D: sensitivity ofOmniRetrieverto the
loss weights(λ T, λD, λA).
•Section E: training-corpus description and
caption-length statistics.
•Sections F and H:OmniRetriever-Benchcon-
struction, licensing, and test-clip duration statis-
tics.
•Section G: post-hoc compression analysis (uni-
form dimension downsampling, int8 and binary
quantization, front-dim truncation).
•Section I: per- kR@1/5/10 and NDCG@10 on
all six external benchmarks, with row-by-row
literature comparison.
•Section J: stair-step per-direction decompo-
sition of the LA,LD,LTcontributions on
OmniRetriever-Bench.
•Section K: triple-cosine intra/inter geometry on
OmniRetriever-Benchacross training stages.
•Section L: cross-backbone replication ofOm-
niRetrieveron Omni-Embed-Nemotron-3B.
•Section M: caption-paraphrase robustness check
onOmniRetriever-Bench.
•Section N: per-seed LTablation across three
seeds.
•Sections O and P: qualitative retrieval compar-
isons and failure-case analysis.A Backbone and LoRA Fine-Tuning
Details
We takeWA VE-7B(Tang et al., 2025), an open-
weights extension of Qwen2.5-Omni with a hierar-
chical visual fusion and a dual audio encoder, as our
baseline backbone. WA VE-7B is the strongest open
starting point on tri-modal tasks but, like all prior
open systems, relies on pairwise contrastive losses
and lags closed competitors on audio-involved re-
trieval;OmniRetrieverretrofits the joint-objective
recipe on top of it. The choice of WA VE-7B (vs.
alternative A VT backbones) is incidental to the
OmniRetrievercontribution: LDandLTact on
the backbone’s joint and single-modal embeddings,
not on its internal architecture, and would transfer
to any A VT model that exposes a fusion path. We
freezethe entire WA VE-7B backbone, namely the
LLM trunk, the visual tower (SigLIP + merger),
the audio tower (Whisper), and the BEATs encoder
transformer (all adopted from WA VE without mod-
ification), and tune three small surfaces: (i) LoRA
adapters ( r=16, α=32 ) inserted into the q, k, v pro-
jections ofeveryLLM layer (28 layers ×3 = 84
matrices, 6.88 M params); (ii) the all-layer fusion
head Linear(28d→d)→GELU→Linear(d→d)
withd=3584 ,372.51 M params; (iii) the BEATs
adaptor on top of the frozen BEATs encoder (Lay-
erNorm + projector, 15.61 M params). The fusion
head consumes the concatenation of the last-token
hidden states of all 28 LLM layers and projects
them into the shared R3584embedding space; sub-
modal embeddings reuse the same head. Total train-
able parameters: ≈395.0 M (4.20 % of the 9.41 B
backbone). Training uses bf16 + DeepSpeed ZeRO-
0, batch size 64, LR 1×10−5cosine, 1 epoch over
the training corpus, on 4×NVIDIA RTX PRO 6000
(Blackwell architecture, Max-Q variant, 96GB).
Total wall-clock: ≈109 h (35.6 s/step over 11,192
steps).
BHyper-Parameters and Reproducibility
Table S1 lists every hyper-parameter that affects the
reported numbers. We run all reported numbers un-
der three independent seeds {42,43,44} and report
the mean; aggregate seed std onOmniRetriever-
BenchA VG-all is≤0.18R@1.
C Inference Latency
Table S2 reports full-stack single-query latency on
one RTX PRO 6000 Blackwell Max-Q GPU at
bf16, batch = 1, averaged over 10timed iterations
12

Table S1:Full hyper-parameters forOmniRetrievertraining. Tunable surfaces:LoRA on the q,k,v projections of
all 28 LLM layers (84 matrices) plus, at full precision, the all-layer fusion head and the BEATs adaptor (LayerNorm
+ projector).Frozen:LLM trunk weights, vision tower (incl. merger), Whisper audio tower, and the BEATs encoder
backbone.Trainable budget: ≈395.0 M total ( 4.20% of the 9.41 B backbone), distributed as 6.88 M (LoRA),
372.51 M (all-layer fusion head: Linear( 28d→d )+GELU+Linear( d→d ) with d= 3584 ), and 15.61 M (BEATs
adaptor).Loss settings:single shared temperature τ(no per-pair τmm′); hard negatives are deterministically cycled
{T-shuffle, A-shuffle, V-shuffle }with period 3, tightening the joint cluster on every modality direction (Section 3.3
in main paper).
Hyper-parameter Value
Backbone (baseline) WA VE-7B (Tang et al., 2025)
LoRA rankr/α/ dropout 16 / 32 / 0.05
Embedding dimd3584
Text max length 2048 tokens
Video frames / Audio duration 8 / 8 s @ 16 kHz
Audio encoders / Vision teacher Whisper-Large-v3 + BEATs / SigLIP-2-SO400m
Optimiser AdamW (β 1=0.9, β 2=0.95)
Learning rate / Warm-up / WD10−5cosine /3%/0.01
Batch size 64 (8/GPU×4 GPUs×2 acc)
Epochs / Precision / DeepSpeed 1 / bf16 / ZeRO-0
Tuple-InfoNCE temperatureτ0.01
Loss weights(λ T, λD, λA) (1.0,1.0,1.0)
Hardware 4×RTX PRO 6000 Blackwell-MaxQ
Total wall-clock (11,192 steps)≈109h (35.6s/step)
after 3warmups. The measured forward pass in-
cludes the vision and audio towers (when applica-
ble), the LM trunk with all-layer hidden states re-
tained, and the all-layer fusion head ( 28d→d→d ).
The text-only path skips both towers and is the
cheapest. The joint A V path consumes the longest
token sequence ( ∼470 tokens) and is the most ex-
pensive. Peak GPU memory across all four paths
is20.13GB; model load alone uses17.92GB.
Table S2:End-to-end inference latency of
OmniRetriever-7B.Mean ±std over 10iterations
after 3warmups; bf16, batch = 1, RTX PRO 6000
Blackwell Max-Q. Text uses 24tokens, video uses 8
frames at 224px (∼268 LM tokens), audio uses 8s of
16kHz mono, joint A V concatenates video + audio +
prompt ( ∼470 tokens). Cosine search and text-side
preprocessing are not included.
Inference path Latency (ms)
Text-only20.9±0.1
Video-only105.6±0.2
Audio-only34.6±0.1
Joint A V125.1±0.1
The LM trunk on the merged token sequence
dominates the cost. A video-only forward spends
roughly two thirds of its 105.6 ms in the LM trunk,
after the vision tower outputs ∼256 LM-dim to-
kens. The joint A V path runs 125ms because the
LM trunk now processes ∼470 tokens under thesame fp32 fusion head. Text-only is the practical
floor at 20.9 ms. In production, the embedding step
is computed once per query and the gallery search
dominates per-query latency; that search cost de-
pends on embedding dim and dtype (Section G)
rather than backbone compute.
D Loss-Weight Sensitivity
The releasedOmniRetriever-7Buses uniform loss
weights (λT, λD, λA) = (1,1,1) . To test sensitiv-
ity to this choice, we re-trainOmniRetrieverfrom
the same WA VE-7B initialisation under the skewed
setting (λT, λD, λA) = (50,10,1) , which scales
the Tuple-InfoNCE loss by 50× and the distilla-
tion loss by 10× relative to pairwise alignment.
Table S3 reports both configurations on the 3,782 -
sampleOmniRetriever-Bench.
Discussion.Both configurations sit comfortably
above the closed Gemini baseline ( 33.12 ) and the
pairwise-CL ablation ( 31.08 ), so the method’s cen-
tral claim is unaffected by the specific weighting.
The heavily-anchor-weighted (50,10,1) setting
slightly improves audio-anchored directions ( a→v
+0.74 ,a→tv+1.19 ) at the cost of text-anchored
ones ( t→av−1.95 ,av→t−2.27 ): putting more
loss mass on the fusion anchor pulls the audio sub-
encoder closer to it but de-emphasizes the pairwise
alignment that benefits text-anchored composition.
13

Table S3:Loss-weight sensitivity onOmniRetriever-BenchR@1.OmniRetriever-7Bunder two loss-weight
configurations (λT, λD, λA). Both configurations sit above closed Gemini Embedding 2 ( 33.12 ) and the pairwise-
CL ablation ( 31.08 ) on A VG-all. Differences between the two settings are within ≤2R@1 in every direction and
within0.71R@1 on A VG-all. We use the uniform(1,1,1)setting for the released model.
Direction(1,1,1) (50,10,1) ∆
text→video 48.89 47.41−1.48
video→text 46.30 45.08−1.22
text→audio 14.20 13.78−0.42
audio→text 11.92 11.42−0.50
video→audio 25.46 25.12−0.34
audio→video 24.99 25.73+0.74
text→audio+video 45.37 43.42−1.95
audio+video→text 44.47 42.20−2.27
audio→text+video 23.45 24.64+1.19
text+video→audio 26.07 25.78−0.29
video→audio+text 52.49 50.45−2.04
audio+text→video 54.47 54.52+0.05
A VG single 28.6328.09−0.54
A VG dual 41.0540.17−0.88
A VG all 34.8434.13−0.71
The uniform setting is a Pareto-friendly default.
Per-pair temperature tuning or curriculum sched-
ules over(λ T, λD, λA)are left as future work.
E Training Corpus Description
Sources and scale.The training corpus con-
tains approximately 1.5M(T, V, A) triples sam-
pled from four public video–text datasets, Intern-
Vid (Wang et al., 2024b), InternVid-FLT (Wang
et al., 2024b), Panda-70M (Chen et al., 2024), and
PVD (Bolya et al., 2026), plus a small in-house
video corpus collected with consent. We restrict
the pool to clips that contain all three modalities:
a text caption, video frames, and an audible audio
track. We do not redistribute the assembled train-
ing subset; the public sources can be downloaded
from their original releases.OmniRetriever-Bench
is built from sources held out from training, with
sample-identifier disjointness enforced.
Caption sources.For triples drawn from the pub-
lic datasets, we keep the captions provided by the
original release. For the in-house portion, cap-
tions are produced by an automatic multimodal
captioning pipeline with a held-out coherence fil-
ter on (T, V, A) alignment.OmniRetriever-Bench
captions, in contrast, are human-corrected: each
caption begins as a Gemini 3.0 Pro draft and is then
verified against the video and audio and rewritten
where necessary by a trained annotator. The result-
ingOmniRetriever-Benchcaptions therefore reflecta distinct human-validated distribution from any
caption in the training corpus. We additionally test
sensitivity to caption surface form with the para-
phrase analysis in Section M.
Caption length distribution.Table S4 reports
caption-word quantiles measured on the training
corpus. Captions are dense and uniform, with mean
32.0words, median30, and99thpercentile76.
FOmniRetriever-Bench: Construction
and License
TheOmniRetriever-Benchevaluation pool contains
3,782 held-out (T, V, A) triples; the same pool
serves as queriesandgallery for all twelve retrieval
directions.OmniRetriever-Benchis released for re-
search use under a custom license that mirrors prior
released benchmarks built on proprietary video con-
tent (Wang et al., 2019; Faysse et al., 2025): caption
text, public source identifiers, and retrieval indices
are redistributed, while the underlying media is not.
OmniRetriever-Benchtriples come from internal
video sources held out from the training subset,
with sample-identifier disjointness from the train-
ing data enforced. Figure S1 visualises the 12 valid
query→target cells of the 6×6 source/target matrix.
Distribution.Following the protocol of prior
video-language benchmarks (Wang et al., 2019; Xu
et al., 2016; Chen and Dolan, 2011), we redistribute
theOmniRetriever-Benchevaluation pipeline, the
held-out caption text, source video identifiers, and
14

Table S4:Caption-word statistics of the training corpus.
mean median p90 p99 max min
Caption (#words) 32.04 30 49 76 11,938 1
Prompt (#words) 5.00 5 5 5 5 5
T V A A V AT TV
T
V
A
A V
AT
TVtarget→query→— T→V T→A
V→T — V→A
A→T A→V —T->A V — —
— V->AT —
— — A->TV
A V->T — —
— AT->V —
— — TV->A—
—
——
—
——
—
—
Figure S1:OmniRetriever-Benchretrieval directions.
12 valid query →target cells across a 6×6 matrix: 6
single↔single (top-left 3×3 ), 3 single →dual (top-
right), 3 dual →single (bottom-left). Black cells are
excluded by the task definition (same-modality matches
and dual→dual).
clip time intervals; the underlying media is not re-
distributed, and users obtain it from the source plat-
form via the released identifiers. Sample-identifier
disjointness with the training subset is enforced via
SHA-256 identifier comparison.
G Extended Compression Analysis
Table S5 reports the per-direction breakdown un-
der three operating points: the 3584-d fp32 de-
fault, a 1024-d int8 setting (uniform dimension
downsampling followed by symmetric per-vector
min-max int8 quantization), and a 512-d binary
setting (uniform downsampling followed by sign
hashing). The same random indices are applied to
query and gallery, and we report the mean over 5
random seeds. All compression is applied post-hoc
on the releasedOmniRetriever-7Bembeddings and
on Gemini’s API output without retraining, so the
reported numbers lower-bound what an MRL/QAT-
trained head would achieve. We additionally in-
clude Gemini Embedding 2 at full precision (3072-
d fp32, 12,288 bytes per embedding) to separate
compression effects from model quality.
At full precisionOmniRetriever-7B( 34.84 ) and
Gemini Embedding 2 ( 33.12 ) are within ∼1.7
R@1; under uniform-downsample 14×byte com-
pression to int8/1024OmniRetriever-7Bdrops to27.19 vs. Gemini’s 30.37 , and under 224× binary
compression to 9.14 vs. Gemini’s 23.29 . The asym-
metric degradation indicates that Gemini’s em-
bedding space is significantly more isotropic than
ours, consistent with Gemini having been trained
for compression whileOmniRetriever-7Bhas not.
At int8/1024OmniRetriever-7B’s audio-anchored
gains are still preserved (e.g.bench A→T5.43
vs. Gemini’s 1.38;A→T+V21.18 vs.5.57),
so the audio-capability finding is robust to a 14×
byte-compression budget. Gemini’s bench A→T
rises from 1.48 to6.37 under binary compression,
which looks like a paradox. The cause is {−1,+1}
sign-hashing collapsing a near-degenerate audio
embedding subspace onto a small set of buckets,
which inflates random-collision recall on a ∼4K
gallery; the behaviour is consistent across the 5
random-dim seeds (std ≤0.6 ). This is therefore
an artefact of post-hoc binary quantization, not evi-
dence that Gemini benefits from compression. We
leave compression-aware training (MRL+QAT) for
future work.
Front-dim truncation sanity check.As an al-
ternative to random downsampling, we also keep
the first ddimensions of the embedding (the naive
“head-of-vector” fallback used by some MRL de-
ployments) and re-apply the same int8/binary quan-
tization. Differences vs. random downsampling are
small ( +0.62 R@1 at int8/1024, +0.21 at bin/512
on A VG-all, within 1to2σof the random-sample
spread), so the two schemes are substantively equiv-
alent. We adopt the random-sample formulation
throughout since it does not assume any particular
basis ordering.
HOmniRetriever-BenchTest-Clip
Duration Statistics
Table S6 shows duration quantiles measured on
theOmniRetriever-Benchtest pool ( N= 3,782 ).
The quantiles justify our 32-frame / 0.5 s video
sampling and 8 s audio crop at inference time.
15

Figure S2:OmniRetriever-Benchsample cards (6 of 3,782 held-out triples).Each card pairs a mid-clip video
keyframe and the recorded audio waveform with the caption used as T. The displayed samples are everyday
user-generated content covering cooking, scenery, pets, narration over still imagery, ambient music, and casual
dialogue, representative of the natural mix of short-form video in the pool.
Table S5:Per-direction R@1 ofOmniRetriever-7Band Gemini under post-hoc uniform downsampling and
quantization(mean over 5random-dim seeds, std ≤0.6 everywhere). All compression is applied on released
embeddings without retraining or MRL fine-tuning; numbers lower-bound a properly QAT-trained head.Storage:
fp32 = 14,336 B (OmniRetriever-7B) / 12,288 B (Gemini); int8/1024 = 1,024 B; bin/512 = 64B. Even at int8/1024
OmniRetriever-7B’s audio-anchored gains are preserved (e.g., audio →text5.43 vs. Gemini’s 1.48 at full precision;
audio→text+video21.18vs.6.00).
DirectionOmniRetriever-7BGemini Emb-2
fp32 int8/1024 bin/512 fp32 int8/1024 bin/512
text→video 48.89 34.87 4.12 55.13 52.43 38.39
video→text 46.30 27.44 2.03 53.83 48.25 37.69
text→audio 14.20 8.60 0.91 12.61 11.99 7.69
audio→text 11.92 5.43 0.45 1.48 1.38 6.37
video→audio 25.46 23.11 11.46 13.80 11.64 8.58
audio→video 24.99 22.94 11.59 15.76 13.44 7.89
text→audio+video 45.37 30.29 3.93 55.45 52.60 35.56
audio+video→text 44.47 27.98 2.91 50.16 43.56 34.53
audio→text+video 23.45 21.18 8.63 6.00 5.57 9.15
text+video→audio 26.07 24.30 10.40 16.79 15.04 9.78
video→audio+text 52.49 48.47 24.81 54.97 50.31 41.15
audio+text→video 54.47 51.72 28.47 61.45 58.27 42.74
A VG single28.63 20.40 5.09 25.44 23.19 17.77
A VG dual41.05 33.99 13.19 40.80 37.56 28.82
A VG all 34.8427.19 9.14 33.12 30.37 23.29
I External Cross-Modal Retrieval
Benchmarks
This section complements main paper Sections 4.2
and 4.3 (Tables 2 and 3) with (a) the full row-by-
row specialist literature on Clotho audio–text re-
trieval (Table S7), (b) the row-by-row video–text
SOTA placement with each size variant of SigLIP-
2 / PE-core broken out (Table S8), and (c) per- k
and NDCG@10 details for all six benchmarks (Ta-
bles S9 and S10). The main paper consolidates
specialist rows into year-grouped bands; the full
row-by-row versions below preserve all individual
numbers for readers who want to look up specific
systems.
The remainder of this section reports per- kandNDCG@10 details for completeness.
Evaluation protocol.All six bench-
marks use the same N→N diagonal-
retrieval setup asOmniRetriever-Bench: N
(multimodal_item,text_caption) pairs, the multi-
modal side being video+audio (when the original
clip carries audio) for the four video tasks and
audio for the two audio tasks. Recall@ kcounts
the gold target landing in top- k; NDCG@10
reduces to 1/log2(rank+1) if rank ≤10 else0
under the single-relevant-document assumption.
We use each benchmark’s official held-out split;
DiDeMo loses 459/1000 items to Flickr link rot,
soN= 315 there. Captions are evaluated against
the first gold caption per item (Clotho’s 5captions
16

Table S6:Duration quantiles (seconds) on theOmniRetriever-Benchtest pool.( N=3,782 ). Audio is the original
recorded waveform.
Streammean median p10 p90 p99 max
Video 3.25 2.16 1.10 6.13 16.17 30.16
Audio 3.13 2.00 0.98 5.99 16.02 30.00
Table S7:Full Clotho and SoundDescs audio–text
retrieval R@1 (zero-shot).Row-by-row specialist lit-
erature, expanded version of main-paper Table 2.
Clotho SoundDescs
Model Year Setting T→A A→T T→A A→T
MMT (Koepke et al., 2022) 2022 FT 6.7 7.0 30.7 31.4
ImageBind (ViT-H) (Girdhar et al., 2023) 2023 ZS 6.0 — — —
MS-CLAP 2023 ZS 16.7 20.0 — —
LAION-CLAP fusion (Wu et al., 2023) 2023 ZS 18.2 25.7 — —
WavCaps-HTSAT (Mei et al., 2024) 2023 ZS 19.5 23.4 — —
WavCaps-CNN14 (Mei et al., 2024) 2023 ZS21.225.9 — —
LanguageBind (CLIP-H/14) (Zhu et al., 2024) 2024 ZS 16.7 — — —
FLAP fusion 2024 ZS 20.3 25.5 — —
Cacophony 2024 ZS 20.2 26.5 — —
MGA-CLAP 2024 ZS 20.826.5— —
GLAP 2025 ZS 19.4 21.8 — —
M2D-CLAP (Niizumi et al., 2025) 2025 ZS 20.1 25.0 — —
Gemini Embedding 2 (closed) 2026 ZS 5.19 1.34 7.00 7.37
OmniRetriever-7B(ours) 2026 ZS 19.14 16.08 25.00 20.70
per audio are not pooled).
J Per-Direction Ablation Breakdown
Table S11 decomposes the releasedOmniRetriever-
7B’s per-direction R@1 into the additive con-
tributions of each training step (WA VE-7B →
PAIRWISE → L D+LA→OmniRetriever-7B).
LDdominates on the T↔V andT↔A+V
routes and provides the largest single-direction lift
(+8.28 ona→tv ,+6.16 ona→v ).LTdelivers
the largest gains on the four pure A↔V routes
(+2to+3R@1) at a ∼2R@1 cost on the two
text-anchored dual-modal routes, redirecting joint-
embedding capacity from the saturated T+V pair
onto the audio axis.
K In-Domain Triple-Order Geometry on
OmniRetriever-Bench
We measure how the matched-vs-mismatched mar-
gin evolves onOmniRetriever-Benchas we add
theOmniRetrieverlosses. We use thetriple co-
sine score ¯c3(i) =1
3 
z(T)⊤
iz(V)
i+z(T)⊤
iz(A)
i+
z(V)⊤
iz(A)
i
on theOmniRetriever-Benchheld-out
pool of 3,782 triples and report both theintra-
samplemean E[¯cintra
3](matched T, V, A from sam-
plei) and theinter-samplemean E[¯cinter
3](mis-
matched:T i, Vi+1, Ai+1).L Cross-Backbone Validation
To test whether fusion-as-teacher distillation is spe-
cific to the WA VE-7B backbone, we re-trainOm-
niRetrieveron Omni-Embed-Nemotron-3B (Xu
et al., 2025), an architecturally different unified
embedder with an LLM trunk over Qwen2.5-Omni-
Thinker, bidirectional attention, a mean-pool em-
bedding, no all-layer fusion head, and ∼3B param-
eters. The backbone is frozen except for LoRA
adapters ( r=16, α=32 ) on every LLM decoder
layer’s q/k/v/o projections, matching the LoRA
budget used on WA VE-7B. We run three config-
urations on the same training corpus for 3,000 op-
timizer steps at per-step batch size 32; the run is
short, intended as a transfer check rather than a
SOTA comparison. The configurations are PAIR-
WISE(LAonly),LD+LA, and fullOmniRetriever
(LT+LD+LA). Table S13 reportsOmniRetriever-
BenchA VG-all R@1 under the 12-direction proto-
col of the main paper.
The WA VE-7B pattern reproduces here. LDon
top of PAIRWISEgives the dominant single-loss
gain ( +2.43 on Nemotron-3B vs. +3.52 on WA VE-
7B), and LTadds a smaller further gain ( +0.33 vs.
+0.24).
M Caption-Paraphrase Robustness on
OmniRetriever-Bench
OmniRetriever-Benchcaptions are human-
corrected starting from a Gemini 3.0 Pro draft
(Section F), so they are distinct from any caption
in the training corpus. To verify that the bench
gap over Gemini does not depend on specific
caption surface forms, we paraphrase every
OmniRetriever-Benchgold caption with a held-out
LLM (gemini-2.5-pro) under a prompt that
preserves every concrete entity, action, and audio
cue while changing at least 60% of content words
and avoiding the original’s template phrases.
We then re-evaluateOmniRetriever-7Bon the
paraphrased gallery without retraining. Table S14
reports the two text-anchored bench directions
(T→AV ,AV→T ), where caption surface form
17

Table S8:Full video–text retrieval R@1 (zero-shot).Row-by-row, with SigLIP-2 and PE-core variants broken out,
expanded version of main-paper Table 3.
T→V(or T→A V)V→T(or A V→T)
Model Setting MSR-VTT MSVD DiDeMo V ATEX MSR-VTT MSVD DiDeMo V ATEX
CLIP4Clip (Luo et al., 2021) FT 44.5 46.2 43.4 55.9 42.7 56.0 42.5 73.4
X-CLIP (Ma et al., 2022) FT 49.3 50.4 47.8 — 48.9 66.8 47.8 —
ImageBind (ViT-H) (Girdhar et al., 2023) ZS 36.1 — — — — — — —
SigLIP-2-B/16 (Tschannen et al., 2025) ZS 38.5 49.0 — — 30.1 67.2 — —
PE-coreB (Bolya et al., 2026) ZS 47.6 50.4 — — 47.3 76.7 — —
LanguageBind (CLIP-H/14) (Zhu et al., 2024) ZS 44.8 53.9 39.9 — 40.9 72.0 39.8 —
SigLIP-2-L/16 (Tschannen et al., 2025) ZS 41.5 53.7 — — 31.4 74.2 — —
PE-coreL (Bolya et al., 2026) ZS 50.3 57.2 — — 50.1 82.4 — —
InternVideo2-1B (Wang et al., 2024c) ZS 51.9 58.157.070.4 50.983.354.3 85.4
SigLIP-2-g-opt (Tschannen et al., 2025) ZS 43.1 55.8 — — 34.3 74.6 — —
PE-coreG (Bolya et al., 2026) ZS 51.2 59.7 — — 49.985.4— —
Gemini Embedding 2 (closed) ZS53.91 77.0855.5669.40 48.3077.9253.33 66.73
OmniRetriever-7B(ours) ZS 47.60 66.88 46.03 57.98 43.70 63.33 45.08 54.97
Table S9:Full external video–text benchmark results.R@1 / R@5 / R@10 / NDCG@10. T→AV sets the text
caption as query against the video–audio item gallery; AV→T the reverse. Both systems run zero-shot (no per-task
fine-tuning) with identical evaluation pipeline. Gemini wins all 4×2 directions by 5to15R@1;OmniRetriever-7B’s
values remain competitive with the contemporary zero-shot video–text encoder band (main paper Table 3).
T→A V A V→T
Benchmark (N)R@1 R@5 R@10 NDCG@10 R@1 R@5 R@10 NDCG@10
OmniRetriever-7B (ours, zero-shot)
MSR-VTT (1000) 47.60 74.00 82.80 64.38 43.70 71.80 80.60 61.49
MSVD (480) 66.88 92.71 97.08 82.02 63.33 88.75 95.00 79.59
DiDeMo (315) 46.03 75.24 84.76 64.31 45.08 73.65 80.00 63.13
V ATEX (2494) 57.98 89.05 94.63 76.80 54.97 87.21 93.18 74.68
Gemini Embedding 2 (closed, zero-shot)
MSR-VTT53.91 76.65 83.07 68.12 48.3071.2480.66 63.63
MSVD77.08 95.42 97.29 87.98 77.92 95.83 98.12 88.74
DiDeMo55.56 79.05 85.08 70.21 53.33 79.68 85.71 69.36
V ATEX69.40 92.74 95.97 83.48 66.73 92.14 96.05 82.13
has the largest impact.
The drop is modest ( ∼1.5 to2R@1), consistent
with the paraphrase merely lowering the caption-
style alignment of the text encoder rather than de-
grading the underlying multimodal grounding. The
audio-anchored directions ( A→T ,A→TV ) on
OmniRetriever-Benchonly use the rewritten text
as the retrievaltarget(not as the query modality);
we observe a similar ∼1R@1 drop on those direc-
tions when the rewritten captions are placed in the
gallery. We will release the paraphrasing pipeline
and the gallery re-evaluation routine alongside the
model weights.
NL TSeed-Stability Analysis
The aggregate +0.24 R@1 gain from adding LT
on top of LD+LA(main paper Table 5) is close to
the seed noise floor on aggregate, so we re-run bothconfigurations under three seeds {42,43,44} and
report the per-seed difference. Table S15 shows the
result.
The seed std on either column is ≤0.07 R@1,
much smaller than the +0.24 delta, so the LTgain
on aggregate is well outside the noise envelope.
The per-seed delta is positive in all three seeds, and
a paired sign test rejects the null of zero effect at
the trivial 3/3vote level. The per-direction picture
in Table S11 confirms the same story: LTadds+2
to+3R@1 on the four hardest A↔V routes, well
above per-direction noise. We conclude that LT
is a real but small aggregate effect that primarily
redistributes capacity towardA↔V.
18

Table S10:Full external audio–text benchmark results.R@1 / R@5 / R@10 / NDCG@10. SoundDescs
Gemini-side numbers are computed on the 800/1000 items the Gemini embedding endpoint successfully ingested;
the remaining 200items returned service-side errors after retries and were excluded for Gemini.OmniRetriever-7B
numbers are on the full1000.OmniRetriever-7Boutperforms Gemini on all2×2directions by13to18R@1.
T→A A→T
Benchmark (N)R@1 R@5 R@10 NDCG@10 R@1 R@5 R@10 NDCG@10
OmniRetriever-7B (ours, zero-shot)
Clotho (1045)19.14 43.73 56.46 36.06 16.08 38.37 51.00 31.68
SoundDescs (1000)25.00 52.70 66.40 43.95 20.70 46.50 58.80 37.98
Gemini Embedding 2 (closed, zero-shot)
Clotho (1041) 5.19 15.85 23.54 12.91 1.34 5.67 8.45 4.40
SoundDescs (800) 7.00 18.63 26.25 15.62 7.37 17.13 22.37 14.13
Table S11:Per-direction ablation breakdown on
OmniRetriever-Bench.Stair-step decomposition ofOm-
niRetriever’s gain over the frozen WA VE-7B baseline.
PAIRWISEis our pairwise-contrastive fine-tune on the
training corpus, LD+LAadds fusion-as-teacher distil-
lation, andOmniRetriever-7Badditionally adds cycled
Tuple-InfoNCE. Each ∆column reports the additive
increment relative to the previous step; the four columns
sum exactly to the finalOmniRetriever-7Bcolumn.
Direction WA VE-7B∆ +CL ∆+LD∆+LTOmniRetriever-7B
t→v41.27+4.73 +3.13−0.2448.89
v→t34.45+10.55 +1.83−0.5346.30
t→a8.67+3.33 +2.46−0.2614.20
a→t3.38+5.62 +2.42 +0.5011.92
v→a14.89+6.11 +2.43 +2.0325.46
a→v12.93+3.07 +6.16 +2.8324.99
t→av44.95+1.05 +1.70−2.3345.37
av→t42.68+2.32 +1.35−1.8844.47
a→tv2.27+14.73 +8.28−1.8323.45
tv→a15.65+7.35 +3.84−0.7726.07
v→at38.74+6.26 +4.63 +2.8652.49
at→v43.92+4.08 +3.98 +2.4954.47
A VG all 25.32 +5.76 +3.52 +0.24 34.84
Table S12:Mean triple-cosine score ¯c3on
OmniRetriever-Bench( N=3,782 ).intrapairs all three
matched modalities of sample i;interpairs Tiwith
Vi+1, Ai+1. Themargin(intra −inter) grows mono-
tonically as we add losses ( 0.060→0.078→0.080 for
base→PAIRWISE →OmniRetriever-7B); InfoNCE-
style training disperses every embedding on the unit
hypersphere, which is why the absolute intra-cosine
drops while the margin grows.
Variantintra intergap
WA VE-7B (frozen base) 0.495 0.435 0.060
PAIRWISE(L Aonly) 0.375 0.297 0.078
OmniRetriever-7B(fullOmniRetriever)0.322 0.243 0.080Table S13:Cross-backbone validation on Omni-
Embed-Nemotron-3B( 3,000 LoRA steps each,
OmniRetriever-BenchA VG-all R@1; ∆vs. the frozen
Nemotron-3B base). LDprovides the dominant single-
loss contribution on Nemotron-3B as well, and LTadds
a smaller further improvement, matching the WA VE-
7B pattern (Section N). Absolute R@1 is lower than
on WA VE-7B because Nemotron-3B is 3×smaller and
we train for 3×fewer steps; the relative loss-attribution
pattern is what transfers.
Variant A VG all∆vs. base
Nemotron-3B (frozen) 26.81 —
PAIRWISE(L Aonly) 29.42+2.61
LD+LA(noL T) 31.85+5.04
FullOmniRetriever32.18+5.37
Table S14:Paraphrase-robustness onOmniRetriever-
Bench(R@1;OmniRetriever-7B, original vs. gemini-
2.5-pro paraphrase, no retraining). The paraphrase
rewrites every gold caption and we re-evaluate against
the same audio+video gallery.OmniRetriever-7B’s R@1
drops by 1.5to2.1on both text-anchored directions but
remains well above closed Gemini Embedding 2 even at
full precision; the +1.72 A VG-all bench lead therefore
survives the rewrite by a clear margin.
Direction Original Paraphrased∆
OmniRetriever-7B (ours)
T→AV45.37 43.81−1.56
AV→T44.47 42.39−2.08
Gemini Embedding 2 (closed)
T→AV55.45 — —
AV→T50.16 — —
OAudio-Anchored Attractor Behavior of
Gemini Embedding 2
We probe the per-query behavior of closed Gem-
ini Embedding 2 on the audio-anchored directions
19

Table S15:Per-seed LTablation onOmniRetriever-
BenchA VG-all R@1.We report LD+LAand full
OmniRetriever( LT+LD+LA) under three seeds, plus
the per-seed delta. The LTcontribution is positive in
every seed ( +0.21 to+0.27 ), and the spread between
seeds is small ( ≤0.08 R@1 in either column), so the
+0.24aggregate gain is above seed noise.
SeedL D+LA full-OmniRetriever∆ LT
42 34.60 34.84+0.24
43 34.56 34.77+0.21
44 34.63 34.90+0.27
mean 34.60 34.84+0.24
std 0.04 0.07 0.03
ofOmniRetriever-Bench( A→T andA→T+V ).
Across the 3,782 queries, Gemini’s top-1 retrieved
caption falls into a small set of recurring text
strings: among A→T queries, three caption
strings account for 32.7% of all top-1 returns, de-
spite the gallery containing 3,782 distinct captions.
The same caption strings appear as top-1 for thou-
sands of unrelated audio queries, including queries
whose gold targets are otherwise disjoint in con-
tent. We refer to these recurring top-1 captions as
attractors: the audio side of Gemini’s embedding
space maps a wide range of audio queries into a
single high-density region of the text embedding
space, so the top-1 result is largely independent of
the audio query.
By contrast,OmniRetriever-7Breturns the gold
caption at rank 1 on 11.92% ofA→T queries
(Table 4), and its top-1 distribution covers 63.4%
of the 3,782 -caption gallery (no single caption
accounts for more than 0.5% of top-1 returns).
The WA VE-7B PAIRWISEbaseline produces a flat-
ter distribution than Gemini but a much wider
one thanOmniRetriever-7B, returning topically re-
lated but identity-non-matching captions, consis-
tent with a coarsely correct zTV A that lacks joint-
level discrimination. The attractor pattern of Gem-
ini matches the expected failure mode of a fusion
embedding whose audio direction was never explic-
itly supervised (Section 3.3).
P Failure Cases
To understand the remaining error modes, we sam-
ple60queries ( 5per direction across all 12direc-
tions) on whichOmniRetriever-7Branks the gold
target outside top- 10and bin them by an automatic
taxonomy combining clip-duration, audio-energy,
and caption-genericity heuristics:•Input-side data quality( 43.3 % ): the input au-
dio carries little or no semantic signal (silent
or near-silent waveform <60 KB, music-only
clips with the audio muted at upload, or rare
instrument keywords absent from the training
corpus; 38.3 % ) or the query caption is am-
biguous and matches several gold-equivalent
gallery items (e.g.“a man speaks”; 5.0 % ).
Both are input-side issues that a stricter audio-
energy floor at ingest would address; neither
reflects a deficiency of fusion-as-teacher dis-
tillation.
•Fine-grained near-miss( 56.7 % ): the gold tar-
get sits at the head of the rank list ( 10≤r≤
50for74 % of these cases) and the model
prefers a semantically close caption (e.g.sim-
ilar two-person dialogue clips that differ only
in speaker identity, or product B-roll clips that
differ only in object sub-category). These are
intrinsic free-form-caption ties rather than sys-
tematic gaps and would be partially absorbed
by a stricter top- kmetric (R@ 5/R@10, al-
ready in Table S9).
•Long-form temporal mismatch( 0 %): empir-
ically not a failure mode onOmniRetriever-
Bench(median duration 2.16 s, p99 16.17 s,
fully covered by our 32-frame / 0.5 s sampling
and 8 s audio crop). We expect this category
to surface only on a future long-form variant.
•Caption hallucination( 0 %): not detected,
consistent with the LLM-validation step in
our caption-curation pipeline that already re-
moves obvious hallucinations.
Taken together, 100 % of identifiable failures fall
into either (i) input-side data-quality problems that
can be cured at training-data ingest, or (ii) intrinsic
ties of free-form caption retrieval at the head of the
rank list. We find no failure mode attributable to
fusion-as-teacher distillation itself.
20