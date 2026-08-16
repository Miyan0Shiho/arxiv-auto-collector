# QV-PIC: Query-Aware Visual Position-Independent Caching for Efficient RAG Serving

**Authors**: Yilin Liu, Rui Meng, Wangze Ni, Jianxin Yan, Heng Cao, Libin Zheng, Peng Cheng, Jinfei Liu

**Published**: 2026-08-12 14:40:43

**PDF URL**: [https://arxiv.org/pdf/2608.12121v1](https://arxiv.org/pdf/2608.12121v1)

## Abstract
Retrieval-Augmented Generation (RAG) repeatedly prefills identical text chunks across queries, incurring redundant computations. Position-Independent Caching (PIC) mitigates it by reusing precomputed Key-Value (KV) across positions, but its efficiency is constrained by the large volume of text tokens. Rendering text chunks as images can compress the text into fewer visual tokens, but the rendered-image PIC suffers more severe quality degradation than the text PIC. This representation-specific gap primarily arises from contextual mismatches across independently compiled caches and the loss of fine-grained textual evidence during visual compression. Existing PIC repair methods mainly address the former through selective recomputation, but they incur online computation and cannot recover lost textual details. We propose QV-PIC, a query-aware dual-resolution PIC reuse framework guided by model-native templates. Offline, QV-PIC compiles visual caches under the model's native chat-template prefix, improving PIC quality without online recomputation. Online, it preserves global context with low resolution and restores fine-grained textual evidence within a high-resolution budget by cumulative query relevance scores, retaining the efficiency benefit of visual compression. Across six tasks, QV-PIC improves average F1 by 21.6 points over vanilla rendered-image PIC, closes the gap to vanilla text PIC, and surpasses optimized text PIC by 2.58 F1 while reducing TTFT by 17.2\%. Relative to full prefill, it cuts TTFT by 83.8%.

## Full Text


<!-- PDF content starts -->

QV-PIC: Query-Aware Visual Position-Independent Caching for Efficient
RAG Serving
Yilin Liu1, Rui Meng2, Wangze Ni1*, Jianxin Yan1,
Heng Cao3, Libin Zheng4, Peng Cheng5, Jinfei Liu1
1Zhejiang University
2Department of Statistics and Data Science, Beijing Normal-Hong Kong Baptist University
3Microsoft
4Sun Yat-sen University
5Tongji University
Abstract
Retrieval-AugmentedGeneration(RAG)repeatedly
prefills identical text chunks across queries, incur-
ringredundantcomputations.Position-Independent
Caching (PIC) mitigates it by reusing precomputed
Key-Value (KV) across positions, but its efficiency
is constrained by the large volume of text tokens.
Rendering text chunks as images can compress the
textintofewervisualtokens,buttherendered-image
PIC suffers more severe quality degradation than
the text PIC. This representation-specific gap pri-
marily arises from contextual mismatches across
independentlycompiledcachesandthelossoffine-
grainedtextualevidenceduringvisualcompression.
ExistingPICrepairmethodsmainlyaddressthefor-
merthroughselectiverecomputation,buttheyincur
online computation and cannot recover lost textual
details. We propose QV-PIC, a query-aware dual-
resolution PIC reuse framework guided by model-
native templates. Offline, QV-PIC compiles visual
caches under the model’s native chat-template pre-
fix, improving PIC quality without online recom-
putation. Online, it preserves global context with
low resolution and restores fine-grained textual ev-
idence within a high-resolution budget by cumula-
tivequeryrelevancescores,retainingtheefficiency
benefitofvisualcompression.Acrosssixtasks,QV-
PICimprovesaverageF1by21.6pointsovervanilla
rendered-image PIC, closes the gap to vanilla text
PIC, and surpasses optimized text PIC by 2.58 F1
while reducing TTFT by 17.2%. Relative to full
prefill, it cuts TTFT by 83.8%.
Introduction
Retrieval-Augmented Generation (RAG) augments user
queries for Large Language Models (LLMs) with
retrieved external documents to support knowledge-
intensive tasks (Lewis et al. 2020; Guu et al. 2020;
Karpukhin et al. 2020; Borgeaud et al. 2022). In long-
document RAG, identical document chunks are repeat-
edly prefilled across queries, incurring redundant pre-
fill computation. Dynamic retrieval rearranges retrieved
chunks with different contextual positions, preventing
∗Corresponding author.
Figure1:Rendered-imagePICusesfewerKVtokensbut
incurs a larger full-prefill-to-PIC quality drop than text
PIConGlyphacrosssixLongBenchQAtasksat72DPI.
efficient reuse of caches that rely on exact prefix match-
ing or predefined structures (Gim et al. 2024; Jin et al.
2025;Zhengetal.2024).Position-IndependentCaching
(PIC) addresses this limitation by compiling reusable
chunks independently and composing their Key-Value
(KV) caches at serving time, enabling cross-position
reuse of repeated content. Existing RAG-oriented PIC
methods (Yao et al. 2025; Hu et al. 2025) are predomi-
nantlytext-based.Althoughrepeatedprefilliseliminated,
the transmission and computation costs associated with
KV caches scale with context length (Kwon et al. 2023;
Liu et al. 2024; Qin et al. 2025a).
Visual-text compression offers a complementary op-
portunity to shorten reusable context representations.
By rendering text chunks as compact images, Vision-
Language Models (VLMs) can encode multiple textual
units into one visual token, increasing per-token infor-
mation density (Li, Lan, and Zhou 2025; Xing et al.
2025;Wei,Sun,andLi2025).WerefertoPICoverren-
dered images asrendered-image PIC, in contrast totext
PICover the original text chunks. Glyph (Cheng et al.
2025), for example, achieves 3-4×token compression
whileretaininglong-contextperformancecomparableto
similarlysizedtext-onlyLLMs.Thisresultsuggeststhat
rendered images can substantially shorten the reusable
KV sequence while preserving useful document infor-
mationunderfullprefill,motivatingrendered-imagePIC
reuse across queries for efficient RAG serving.
arXiv:2608.12121v1  [cs.CL]  12 Aug 2026

This opportunity raises a core question for efficient
RAG serving:can rendered-image PIC match the
reuse quality of text PIC?Under identical conditions,
we compare text and rendered-image PIC using Glyph
across six LongBench Question-Answering (QA) tasks.
As shown in Figure 1, although rendering substantially
reducestokencount,rendered-imagePICsuffersgreater
degradationfromfullprefillthantextPIC.Thisindicates
thatrenderedimagescannotdirectlyinheritthePICreuse
capabilityoftext.Thisrepresentation-dependentgapre-
flects two coupled failure modes. First, independently
compiled caches lack the contextual conditions avail-
able during full prefill, causing cache-state mismatch
after composition. Second, text tokenization preserves
characters and words as discrete symbols, whereas vi-
sualencodingcompressescharacters,digits,punctuation,
and local layout into fewer visual tokens, each aggregat-
ing multiple textual units. Fine-grained answer-bearing
evidence may therefore be blurred, conflated, or omit-
ted before the rendered-image KV cache is constructed.
Rendered-image PIC must consequently address both
compilation-contextmismatchandfine-grainedevidence
loss.
Prior PIC repair methods primarily address the first
failure mode through selective recomputation or chunk-
boundary correction (Yao et al. 2025; Hu et al. 2025;
Zhao et al. 2025b; Qin et al. 2025b). Such methods can
refresh context-dependent states after caches are com-
posed,buttheyintroduceadditionalcomputationintothe
online path and operate on an already fixed visual rep-
resentation. They cannot reconstruct characters, digits,
punctuation, or layout cues that were not preserved dur-
ingvisualencoding.Thislimitationisparticularlyconse-
quential for rendered images, whose answer-bearing in-
formation often lies in fine-grained textual details rather
than the coarse visual semantics sufficient for natural-
image understanding. Consequently, efficient rendered-
image PIC poses two key challenges:
Challenge 1: Efficient and stable repair of
compilation-context mismatch.Prepending disposable
prefix during compilation and stripping it afterward
can absorb chunk-initial attention sinks without online
recomputation. But arbitrary dummy prefixes induce
prefix-dependent cache states, making the repair sensi-
tive to token identity and length.
Challenge2:Efficientrestorationofvisual-textde-
tails.Renderingresolutionnotonlycontrolstextualleg-
ibility but also visual-token count. Uniform low reso-
lution is efficient but risks discarding fine-grained ev-
idence. High resolution preserves richer textual details
butsharplyincreasestokens,erodingtheefficiencygains
of visual-text compression.
In response to these challenges, we proposeQV-
PIC, a model-native template-conditioned and query-
aware dual-resolution framework for rendered-image
PIC, which transforms fixed text chunks into control-
lablefine-grainedunits.ForChallenge1,QV-PICcom-
piles each rendered image under the model-native chat-template prefix and strips the shared-prefix KV entries
beforestorage.Thisnativetemplateprovidestherequest-
invariant prompt-format condition that is present dur-
ing full prefill, reducing systematic compilation-context
mismatchwithoutonlinerecomputation.ForChallenge
2, QV-PIC precompiles low- and high-resolution cache
versions for each rendered image. At serving time, it
begins with complete low-resolution context coverage
and promotes only a bounded set of query-relevant ren-
dered images to high resolution according to cumula-
tive query relevance. The two components are comple-
mentary: template-conditioned compilation first solidi-
fies the quality foundation of the rendered image PIC,
andquery-awaredual-resolutionallocationthenrestores
query-specifictextualdetails.Theonlinepathrequiresno
rendered-image generation, visual encoding, or context-
side full prefill. Our contributions are summarized as
follows:
•Werevealthesignificantimpactoftextandren-
dered image representations on PIC reuse.For
identical text content, rendered-image PIC suffers
more severe degradation than text PIC, but the
former has more potential in quality-latency per-
formance.
•We propose QV-PIC, a template-conditioned
and query-aware dual-resolution framework
forrendered-imagePIC.Itefficientlyreducesthe
compilation-context mismatch and improves fine-
grainedtextualevidencefidelityofrendered-image
PIC.
•We demonstrate that QV-PIC achieves consis-
tentquality-latencyimprovements.QV-PICim-
proves average F1 by 21.6 points over vanilla
rendered-imagePIC,eliminatingits12.2-pointgap
to vanilla text PIC and outperforming optimized
text PIC by 2.58 points while reducing TTFT by
17.2%. Compared with full prefill, it reduces on-
line prefill time by 83.8%.
Background
Position-Independent Caching for Text
In RAG serving, the same text chunk may be retrieved
by different queries with varying prefixes, orders, and
contextualpositions.Conventionalprefixcachingreuses
KV caches only when requests share fixed prefixes or
predefined layouts. For example, Prompt Cache (Gim
et al. 2024) predefines cacheable modules with posi-
tions, while RAGCache (Jin et al. 2025) applies tree-
based KV retrieval and reuses prefix paths across GPU
andmemory.Suchprefix-dependentreuseisdifficultfor
dynamic retrieval, where the same chunk may appear in
different surrounding contexts. Text PIC addresses this
limitation by independently compiling text chunks into
reusable KV caches and linking retrieved caches during
online serving. Cache-Craft (Agarwal et al. 2025) iden-
tifies reusable chunk caches and selectively recomputes

context-sensitive states; EPIC (Hu et al. 2025) formal-
izes PIC as a compile-and-link framework that recom-
putes only leading tokens to mitigate attention sinks;
TurboRAG (Lu et al. 2025) stitches precomputed KV
caches with independent attention masks and reordered
RoPE positions. However, these Text PIC methods still
incur online overhead from token selection, recomputa-
tion, or scheduling. Although static offline linking can
reduce latency, it is constrained by the form and seman-
tics of precompiled prefixes. More importantly, they fail
to shorten text-chunk representations. Thus, even with-
outfullprefill,long-documentRAGstillrequiresloading
andtransferringlargetextKVcaches,limitingPICreuse
efficiency (Liu et al. 2024; Qin et al. 2025a).
Position-Independent Caching for Images
Visual-textcompression renderstextasimages, increas-
ingvisual-tokeninformationdensity.Priorwork,includ-
ingTextorPixels(Li,Lan,andZhou2025),VIST(Xing
et al. 2025), Glyph (Cheng et al. 2025), DeepSeek-
OCR (Wei, Sun, and Li 2025), and DeepSeek-OCR
2 (Wei, Sun, and Li 2026), shows its effectiveness for
long-context modeling and token compression. How-
ever, compressed visual text remains sensitive to ren-
dering resolution and visual encoding, and OCR read-
ability alone cannot reflect long-range retrieval and rea-
soning quality (Zhao et al. 2025a). Recent work adapts
visual processing to task demands: AgentOCR (Feng
et al. 2026) uses segment optical caching to reuse
rendered interaction-history segments, while Agentic-
OCR (Wang et al. 2026) identifies query-relevant re-
gions and performs OCR on demand. These methods
reduce irrelevant visual input but do not reuse page-
level KV caches that can be position-independently as-
sembled across requests. Multimodal caching methods
such as MPIC (Zhao et al. 2025b) and VLCache (Qin
et al. 2025b) further reuse visual intermediate states or
language-model KV caches with selective recomputa-
tion. However, their fixed ordinary resolution may dis-
card fine-grained textual evidence that later KV repair
cannot recover, while recomputation still incurs online
overhead.
Methodology
QV-PIC addresses two sources of degradation in
rendered-image PIC: cache-state mismatch and the loss
of fine-grained textual evidence. Offline, model-native
template-conditioned compilation improves indepen-
dently compiled KV quality and establishes a reliable
reuse basis. Online, query-aware dual-resolution allo-
cation selectively restores query-relevant visual details
without rerendering or re-encoding images.
Framework Overview
Given a queryq, letC(q) = (c 1, . . . , c n)denote the
retrievedchunksinthecurrentcontextorder.Eachchunk
ciis rendered asxr
i= Render(c i;r)at resolutionr.As shown in Figure 2, QV-PIC follows a two-phase
workflow.
Phase I: Template-conditioned cache preparation.
For each reusable chunkc i, QV-PIC renders low-
and high-resolution images and independently compiles
one cache per resolution under the model-native chat-
template prefix. It strips the prefix KV entries before
storage, but retains the resulting template-conditioned
rendered-image KV entries. The cache bank stores both
resolution variants, token-count metadata, source-order
metadata, and a source-text embedding for later routing.
Phase II: Query-aware cache assembly.At serving
time, QV-PIC scores the retrieved chunks against the
query and promotes at mostBquery-relevant rendered
images to high resolution. All retrieved chunks remain
inthecontext,andexactlyonecacheversionisactivated
for each rendered image. The relevance ranking is used
only for resolution allocation; cache assembly follows
the current context order of the RAG request. After M-
RoPE re-anchoring, the assembled cache is supplied as
past KV, so the VLM only computes query prefill and
answer generation online.
Model-Native Template-Conditioned
Compilation
In full prefill, the rendered-image context is processed
under the model-native chat-template prefix, which pro-
vides the request-invariant prompt-format condition of
the VLM’s multimodal serving interface. Prefix-free
compilation removes this condition, while dummy pre-
fixes replace it with arbitrary tokens whose effect varies
with token identity and length. QV-PIC instead uses the
chat-template prefix as the compile-time condition and
strips only its KV entries before storage. This reduces
prompt-format mismatches in independently compiled
rendered-image caches.
LetMbe the VLM andhits model-native chat-
template prefix. For a rendered imagexr
i, QV-PIC con-
structs
Cr
i= Striph(KV M([h;xr
i])),(1)
whereStriphremoves the KV entries corresponding to
h. Although these prefix entries are discarded, the re-
tained rendered-image entries are still computed under
the native template condition. At serving time, QV-PIC
adds only one shared prefix cache,C h= KV M(h).
Since each rendered-image cache is compiled inde-
pendently,itskeysarestoredbeforeM-RoPErotation(Su
et al. 2024; Wang et al. 2024). After the current context
order and resolution assignment are fixed, QV-PIC de-
rivestherequestpositionsP i(q)foreachactivatedcache
and applies
Kr
ℓ,i=R ℓ ¯Kr
ℓ,i,Pi(q)
.(2)
where ¯Kr
ℓ,iis the unrotated key at layerℓ, andR ℓis
themodel-nativeM-RoPEoperator.Valuesareposition-
independentandarestitchedinthesamecurrentcontext
order.Templateconditioningimprovestheindependently

Figure 2: Overview of QV-PIC. Offline, model-native template-conditioned compilation builds low- and high-
resolutionrendered-imagecachesandsource-textembeddings.Online,query-awaredual-resolutionallocationselects
onecacheversionforeachrenderedimage,assemblestheselectedcachesunderthecurrentcontextorder,re-anchors
M-RoPE positions, and prefills only the query.
compiledrendered-imageKVentries,whileM-RoPEre-
anchoring places them at their request positions. Thus,
the two operations are complementary.
Query-Aware Dual-Resolution Allocation
Uniform low resolution reduces visual-token and KV
costs, but may weaken characters, numbers, and local
textualevidence.Uniformhighresolutionpreservesmore
detail,butincreasestheactiveKVsizeforeveryrendered
image.QV-PICthereforeprecompiles bothversionsand
activateshighresolutiononlyforquery-relevantrendered
images:
Bi={CL
i,CH
i},(3)
whereCL
iandCH
idenote the low- and high-resolution
caches of rendered imagei.
Query-relevance scoring.QV-PIC uses a frozen
BGE-M3 encoder (Chen et al. 2024)E(·)to embed the
source text of each rendered image offline and the query
online:
ei=E(ci)
∥E(c i)∥2,e q=E(q)
∥E(q)∥ 2.(4)
The relevance score is cosine similarity:
˜si=e⊤
qei, s i= [˜si]+= max(˜s i,0).(5)
Letπ(q)sorttheretrievedchunksby˜s iindescendingor-
der. This ranking is used only to choose high-resolution
caches. WhenP
isi>0, QV-PIC selects the small-
est top-ranked set whose cumulative positive relevancereaches thresholdα, capped by budgetB:
k⋆= min 
B,min(
k:Pk
j=1sπjPn
i=1si≥α)!
.(6)
Ifallscoresarenon-positive,QV-PICselectsthehighest-
scoringchunkasadeterministicfallback.Thepromoted
set and resolution assignment are
S(q) ={π 1, . . . , π k⋆}, r i(q) =H, i∈ S(q),
L,otherwise.
(7)
Here,˜s imeasures query relevance, whereasr i(q)de-
notes the assigned resolution.
Online assembly and cost.After resolution assign-
ment,QV-PICactivates{Cri(q)
i}n
i=1andassemblesthem
underthecurrentcontextorder.IfnL
iandnH
idenotethe
low-andhigh-resolutiontokencountsofrenderedimage
i, the active rendered-image prefix length is
N(q) =nX
i=1nL
i+X
i∈S(q) 
nH
i−nL
i
,|S(q)| ≤B.
(8)
Thus,high-resolutionoverheadispaidonlyforpromoted
rendered images. Online routing requires query encod-
ing, similarity scoring, and ranking:
Troute =TE(q) +O(nd) +O(nlogn),(9)
wheredis the embedding dimension. Rendering, visual
encoding, and rendered-image KV compilation remain
offline.

Experiments
In this section, we conduct experiments to evaluate QV-
PIC by addressing the following questions:
Q1:Canmodel-nativetemplate-conditionedcompila-
tionimprovethereusequalityofindependentlycompiled
rendered-image caches?
Q2:Undertemplate-conditionedcompilation,howdo
the F1 and TTFT of rendered-image PIC change with
uniform DPI scaling?
Q3:CanQV-PICachievehigheraverageF1withlower
average TTFT than uniform 120-DPI rendered-image
PIC and text PIC?
Q4:HowwelldoesQV-PICgeneralizebeyondGlyph?
Experimental Configuration
ImplementationWe implement all methods in a uni-
fied Hugging Face-PyTorch inference framework and
evaluate them on a server equipped with eight NVIDIA
A800 80 GB GPUs. All methods using the same back-
bone share identical configurations. Following the ren-
deringprotocolofGlyph (Chengetal.2025),wefixthe
rendering canvas size, margins, font, and line spacing
while varying only DPI. We extend Glyph’s 72/96/120-
DPI range to 144 and 168 DPI at 24-DPI intervals. QV-
PIC uses 72/120 DPI as its dual-resolution configura-
tion: 72 DPI preserves full-context coverage at low cost,
whereas 120 DPI provides a clear average quality gain
without the larger token and latency costs of 144 and
168DPI.Forquery-awaredual-resolutionallocation,we
rank rendered images by relevance and select the small-
esttop-rankedsetwhosecumulativenormalizedpositive
relevance reachesα= 0.65, capped atB= 4high-
resolution rendered images. Owing to the compute bud-
get, NarrativeQA is evaluated at 72, 96, and 120 DPI,
whereas the other tasks use the complete DPI sweep.
Model SelectionForQ1-Q3, we use Glyph 9B as
theprimarymodel.Glyphreceivesrendered-text-specific
adaptation through continual pretraining on rendered
long-text data and OCR-aware SFT/RL. This special-
ization reduces confounding from basic rendered-text
recognition, allowing Q1-Q3 to focus on cache com-
pilation and resolution allocation. ForQ4, we evalu-
ate two technically compatible general-purpose VLMs
of comparable scale: GLM-4.1V-9B-Thinking (GLM-V
Team 2025) and LLaVA-OneVision-2-8B-Instruct (An
et al. 2026). Both support multi-image inputs and pro-
videOCRanddocument-understandingcapabilities,but
neitherhasundergonerendered-text-specificadaptation.
GLM-4.1V provides a related-family setting because
Glyph is initialized from GLM-4.1V-9B-Base, whereas
LLaVA-OneVision-2usesadifferentvisionencoder,lan-
guage backbone, and training recipe, providing a cross-
family setting. These models are used as conservative
transfer probes. Positive results on them would indicate
that QV-PIC does not rely entirely on Glyph’s rendered-
text-specific training. Meanwhile, dedicated rendered-
textadaptationmayprovideadditionalqualityheadroomfor QV-PIC on future compatible backbones.
BaselinesForQ1, we compare a prefix-free baseline
with three cache-state repair strategies: dummy-prefix
conditioning usingk∈ {2,4,8,16}repetitions of the
placeholder tokenx, model-native template condition-
ing, and an efficient recomputation method EPIC-2/4
without token selection. Thek= 4dummy prefix
matches the native chat-template length, while the re-
maininglengthstestsensitivitytoarbitraryprefixlength.
ForQ2,wecomparefullprefillandtemplate-conditioned
PIC for both text and rendered-image inputs across DPI
settings. This separates representation quality from PIC
degradation and evaluates the quality and latency ef-
fects of uniform DPI scaling. ForQ3, we compare QV-
PICwithtemplate-conditionedtextPIC,uniform72-and
120-DPIrendered-imagePIC,andQV-PICwithouttem-
plateconditioning.Thisisolatesthecontributionofdual-
resolution allocation and its complementarity with tem-
plateconditioning.Q4repeatsthesamewithin-backbone
comparison on two additional VLMs, measuring gener-
alization relative to each model’s own PIC baseline.
DatasetsForQ1-Q4, we select six long-context
question-answering (QA) tasks from LongBench (Bai
et al. 2024). 2WikiMQA, HotpotQA, and MuSiQue
cover multi-document, multi-hop evidence aggregation.
MultiFieldQA-enandNarrativeQAevaluateevidencelo-
calization and holistic understanding within long single
documents, while TriviaQA focuses on factoid question
answering. Together, these tasks span single- and multi-
document contexts, localized and distributed evidence,
and direct retrieval and multi-hop reasoning, providing
complementarytestsofthecompositionandreuseofin-
dependently compiled rendered-image caches. We use
all 1,150 examples in LongBench evaluation subsets.
MultiFieldQA-en contains 150 examples, and each of
the other five tasks contains 200. Results are first aver-
aged within each task and then equally averaged across
tasks.
MetricsQV-PIC is evaluated by answer quality, on-
line latency, and token size. Answer quality is measured
by official LongBench token-overlap F1 using one de-
terministic run per example. TTFT, averaged over three
runs, is measured from a CUDA synchronization imme-
diately before each online request to first-token logits.
For full prefill, TTFT includes visual encoding when
applicable, full-context prefill, and first-token computa-
tion. For PIC, TTFT includes CPU-to-GPU KV transfer
andmaterialization,cachecomposition,globalpositional
re-anchoring, query-suffix prefill, and first-token com-
putation. QV-PIC additionally includes BGE-M3 query
encoding, relevance scoring, ranking, and resolution as-
signment.
Q1: Effectiveness of Model-Native
Template-Conditioned Compilation
To answer Q1, we compare prefix-free compilation,
dummy-prefix compilation withk∈ {2,4,8,16}, and

model-native template-conditioned compilation. For the
latter two, the prefix is prepended during offline com-
pilation and their KV entries are then discarded, retain-
ingonlytheprefix-conditionedKV.Rendered-imageex-
periments use 72 and 120 DPI. Since the native chat-
template prefix has four tokens, dummy-prefix-4 serves
asalength-matchedcontrol.WealsocompareEPIC-2/4,
which recomputes the first two or four chunk tokens on-
line. Figures 3 and 4 report the six-task average F1.
0 10 20 30 40 50 60 70
Average F1Prefix-free
Dummy prefix (k=2)
Dummy prefix (k=4)
Dummy prefix (k=8)
Dummy prefix (k=16)
EPIC-2
EPIC-4
Template prefix32.7
47.8
45.8
43.6
40.8
33.3
32.2
48.832.9
48.9
46.9
43.8
40.7
32.5
31.5
52.172 DPI 120 DPI
Figure 3: Six-task average F1 of rendered-image PIC
under different cache-compilation and repair settings at
72 and 120 DPI.
Prefix-conditioned compilation outperforms reso-
lution scaling and leading-token recomputation.
Prefix-free rendered-image PIC obtains average F1
scoresof32.7and32.9at72dpiand120dpi,respectively,
whichare12.2and12.0pointslowerthanprefix-freetext
PIC. Increasing DPI provides almost no improvement.
EPIC-2/4achievesF1scoresof31.5and33.3,remaining
comparable to prefix-free Rendered-Image PIC. In con-
trast, the dummy prefixk= 2improves F1 to 47.8 and
48.9 at the two resolutions, indicating that conditioning
each chunk during offline compilation is more effective
than recomputing a few leading tokens of each chunk
online. However, as the dummy-prefix length increases
from2to16,theaverageF1ofbothimageandtextdrops
substantially, indicating its sensitivity to arbitrary prefix
length.Template-conditionedcompilationachievesaver-
ageF1scoresof48.8,52.1,and51.7for72-DPIrendered
images,120-DPIrenderedimages,andtext,respectively.
Itoutperformsthelength-matcheddummyprefixby3.0,
5.2, and 5.1 points. This confirms that the gain comes
from alignment with the model’s learned input interface
rather than the mere presence or length of prefix condi-
tioning.
Answer to Q1. Model-native template conditioning
provides the highest PIC quality across resolutions
and modalities.At 120 DPI, it raises the rendered-
image PIC from 32.9 to 52.1 F1, converting its origi-
nal 12.0-point gap from text PIC into a 0.4-point ad-
vantage. Therefore, model-native template conditioningconstructs higher-quality reusable caches entirely of-
fline,withoutonlinerecomputationortuninganarbitrary
dummy-prefix length.
0 10 20 30 40 50 60 70
Average F1Prefix-free
Dummy prefix (k=2)
Dummy prefix (k=4)
Dummy prefix (k=8)
Dummy prefix (k=16)
Template prefix44.9
49.2
46.6
43.6
39.0
51.7
Figure4:Six-taskaverageF1oftextPICunderdifferent
cache-compilation settings.
Q2: Effects of Uniform DPI Scaling on Quality
and Latency
Q1 shows that DPI alone cannot repair independently
compiled caches, whereas template conditioning estab-
lishes a reliable reuse basis and allows higher resolution
to deliver further average F1 gains. Q2 therefore exam-
ineshowuniformDPIscalingaffectstheF1andTTFTof
rendered-image PIC relative to text PIC and full prefill.
Uniform DPI scaling yields unstable F1 changes
whileTTFTincreasesconsistently.AsshowninFig-
ure 5, the best rendered-image full-prefill configura-
tions approach or match text full prefill across the six
tasks,confirmingthequalitypotentialofrendered-image
inputs. Template-conditioned rendered-image PIC im-
proves from an average F1 of 48.8 at 72 DPI to 52.1 at
120DPI,andatleastoneDPIsettingreachesorexceeds
text PIC on four tasks. However, the per-task F1 gains
are non-monotonic. Increasing DPI may improve, pre-
serve,orreduceF1,showingthatadditionalvisualdetail
does not reliably translate into higher reuse quality. In
contrast, TTFT increases consistently as visual tokens
and KVcaches grow. Althoughincreasing DPI provides
morevisualdetail,italsoaddsvisualtokenswhoseaddi-
tionaldetailisnotconsistentlyusefultothecurrentquery.
Moreover,evenat120DPI,rendered-imagePICremains
roughlyanorderofmagnitudefasterthanrendered-image
full prefill on most tasks.
AnswertoQ2. Templateconditioningenablesmod-
erate DPI increases to improve rendered-image
PIC quality while retaining substantial full-prefill
speedups.However,thegainsbecomelimitedorunstable
whereas TTFT increases consistently, motivating selec-
tivehigh-resolutionallocationtoquery-relevantrendered
images.

0.1 0.20.3 0.50.7 1 23 550556065707580F1
higher F1 ↑
lower TTFT ←2WikiMQA
72 dpi120 dpi168 dpi
72 dpi120 dpi
168 dpi
0.20.3 0.50.71 23 57105055606570
HotpotQA
72 dpi120 dpi
168 dpi
72 dpi120 dpi168 dpi
0.20.3 0.50.71 23 571030354045505560F1
MuSiQue
72 dpi120 dpi168 dpi
72 dpi
120 dpi168 dpi
0.1 0.20.3 0.50.7 1 23 5354045505560
MultiFieldQA-en
72 dpi120 dpi168 dpi
72 dpi120 dpi168 dpi
0.20.3 0.50.7 1 23 571080859095F1
TriviaQA
72 dpi120 dpi168 dpi72 dpi
120 dpi168 dpi
0.3 0.50.7 1 23 5710151520253035
NarrativeQA
72 dpi120 dpi
72 dpi120 dpi
TTFT (s, log scale)Text full prefill
Template-conditioned rendered-image PICRendered-image full prefill
Template-conditioned text PICFigure 5: Per-task F1-TTFT comparison of full prefill
and template-conditioned PIC for text and rendered im-
age across DPI settings. Image points are connected in
ascending DPI order, and TTFT is shown on a logarith-
mic scale.
Q3: Joint F1-TTFT Improvement via
Query-Aware Dual-Resolution Allocation
Q3 examines whether allocating a bounded high-
resolutionbudgettoquery-relevantrenderedimagescan
improve the overall performance of rendered-image PIC
reuse. QV-PIC retains most rendered-image caches at
72DPIandselectsthemostrelevantoneswith120DPI.
Additionally,weremovetemplateconditioningtoevalu-
ate its synergy with query-aware dual-resolution alloca-
tion.
Query-aware dual-resolution allocation improves
F1 without uniform high-resolution overhead.As
shown in Figure 6, QV-PIC simultaneously improves
F1 and reduces TTFT over uniform 120-DPI rendered-
imagePIConHotpotQA,MuSiQue,TriviaQA,andNar-
rativeQA, indicating that enhancing only query-relevant
images preserves useful visual-detail gains while avoid-
ing unnecessary visual overhead on irrelevant pages.
Compared with text PIC, it improves F1 and re-
duces TTFT on MuSiQue, TriviaQA, and NarrativeQA,
achieveshigherF1atcomparableTTFTon2WikiMQA,and maintains similar F1 at lower TTFT on HotpotQA
and MultiFieldQA-en. Moreover, template conditioning
raisestheaverageF1ofprefix-free72-DPIPICfrom32.7
to48.8,whereasdual-resolutionallocationalonereaches
only 32.5. Combining both components increases the
average F1 to 54.3, with gains on all tasks, finally sur-
passing the 51.7 F1 of text PIC. Template conditioning
therefore establishes a reliable KV basis, while query-
awaredual-resolutionallocationprovidesquery-relevant
fine-grained evidence.
0.1 0.3 0.5 0.75055606570F1
2WikiMQA
72120168
0.1 0.3 0.5 0.750556065
HotpotQA
72120168
0.1 0.3 0.5 0.730354045
MuSiQue
72
120168
0.1 0.3 0.5 0.73540455055F1
MultiFieldQA-en
72120168
0.1 0.3 0.5 0.780859095
TriviaQA
72120 168
0.1 0.3 0.5 0.7152025
NarrativeQA
72 120
0.1 0.2 0.3 0.4
TTFT (s)5055F1
72120QV-PIC vs Text PIC
+2.58 F1; −17.2% TTFT
0 20 40 60 80 100
F12Wiki
Hotpot
MuSiQ
Multi.
Trivia
Narr.
MacroTTFT (s)
(a) Per-task F1–TTFT comparisons
(b) Six-task average F1–TTFT comparison (c) Component ablation across tasksTemplate-conditioned rendered-image PIC
Prefix-free rendered-image PIC (72 DPI)
QV-PIC w/o template conditioningQV-PIC
Template-conditioned text PIC
Figure 6: F1-TTFT comparisons on different PIC meth-
ods and component ablation of QV-PIC.
Answer to Q3. QV-PIC preserves the visual com-
pressionadvantagewhileachievingbetteroverallF1.
Compared with text PIC and template-conditioned 120-
DPI rendered-image PIC, QV-PIC attains higher aver-
age F1 with lower average TTFT. The ablation further
confirms that the gains arise from the synergism of
template-conditionedcompilationandquery-awaredual-
resolution allocation.
Q4: Cross-Model Generalization of QV-PIC
Q4 examines whether QV-PIC remains effective
on general-purpose VLMs GLM-4.1V and LLaVA-
OneVision-2. GLM-4.1V provides a related-family set-
ting, whereas LLaVA-OneVision-2 provides a cross-
family test. We compare prefix-free 72 DPI rendered-
imagePIC,template-conditionedrendered-imagePICat
72and120DPI,template-conditionedtextPIC,andQV-
PIC.
QV-PIC consistently strengthens rendered-image
PIC.As shown in Figure 7, template-conditioned 72-
DPI rendered-image PIC substantially improves average
F1overprefix-freecompilationonbothmodels.Uniform
120-DPI compilation further improves F1 on both mod-
els,confirmingthattemplate-conditionedcompilationis
not confined to Glyph’s rendered-text-specific adapta-

tion. On GLM-4.1V, QV-PIC achieves the highest aver-
age F1 while requiring lower TTFT than both uniform
120-DPI rendered-image PIC and template-conditioned
textPIC.OnLLaVA-OneVision-2,QV-PICsubstantially
improves over the uniform 72-DPI configuration and
nearly matches the highest average F1 obtained by uni-
form 120-DPI rendered-image PIC and text PIC, while
requiring markedly lower TTFT than either.
Answer to Q4. QV-PIC shows promising general-
izationbeyondtheprimaryGlyphmodel.Acrossboth
related- and cross-family general-purpose VLMs, it ei-
therachievesthehighestF1withlowerTTFTorretainsa
near-bestF1atalowerTTFT.Thus,itsgainsarenotlim-
itedtoGlyph’srendered-textspecialization,althoughthe
magnitude of the gain depends on the underlying VLM.
0.2 0.4 0.6203040Average F1GLM-4.1V
0.2 0.4 0.6LLaV A-OneVision-2Prefix-free rendered-image
PIC (72 DPI)Template-conditioned
rendered-image PIC (72 DPI)
QV-PIC Template-conditioned
rendered-image PIC (120 DPI)Template-conditioned text PIC
TTFT (s)
Figure 7: Cross-model six-task average F1-TTFT com-
parison of QV-PIC.
Conclusion
Regarding the reuse-quality degradation caused by
cache-state mismatch and fine-grained evidence loss in
rendered-imagePIC,weproposeQV-PIC,aquery-aware
visualPICframeworkforefficientRAGserving.Itcom-
bines model-native template-conditioned cache compi-
lation with query-aware dual-resolution cache assembly
to reduce compilation-context mismatch and preserve
query-relevant textual evidence. Across six LongBench
QAtasks,QV-PICimprovesrendered-imagePICby21.6
F1pointsandsurpassesoptimizedtextPICanduniform
120-DPI rendered-image PIC with lower TTFT. Com-
pared with full prefill, it reduces online prefill time by
83.8%, enabling fast and accurate long-document RAG
with reusable visual-text caches.
References
Agarwal, S.; Sundaresan, S.; Mitra, S.; Mahapatra, D.;
Gupta, A.; Sharma, R.; Kapu, N. J.; Yu, T.; and Saini,
S. K. 2025. Cache-Craft: Managing Chunk-Caches for
EfficientRetrieval-AugmentedGeneration.Proceedings
oftheACMonManagementofData,3(3):136:1–136:28.
An, X.; Xie, Y.; Tang, F.; Yan, Y.; Tan, H.; Zhu, D.;
Chen, C.; Zhao, X.; Qin, B.; Yang, K.; Shen, Y.; Zhang,
Y.; Zhang, K.; Zhang, W.; Cheng, Z.; Zhang, N.; Wu,
C.; Ge, C.; Ran, Z.; Song, D.; Li, C.; Feng, S.; Hu,M.; Chen, Z.; Niu, J.; Li, B.; Feng, Z.; Liu, Z.; Ge,
Z.; and Deng, J. 2026. LLaVA-OneVision-2: Towards
Next-Generation Perceptual Intelligence.arXiv preprint
arXiv:2605.25979.
Bai, Y.; Lv, X.; Zhang, J.; Lyu, H.; Tang, J.; Huang,
Z.; Du, Z.; Liu, X.; Zeng, A.; Hou, L.; Dong, Y.; Tang,
J.; and Li, J. 2024. LongBench: A Bilingual, Multitask
Benchmark for Long Context Understanding. InPro-
ceedings of the 62nd Annual Meeting of the Association
forComputationalLinguistics(Volume1:LongPapers),
3119–3137. Bangkok, Thailand: Association for Com-
putational Linguistics.
Borgeaud,S.;Mensch,A.;Hoffmann,J.;Cai,T.;Ruther-
ford,E.;Millican,K.;VanDenDriessche,G.B.;Lespiau,
J.-B.; Damoc, B.; Clark, A.; De Las Casas, D.; Guy, A.;
Menick, J.; Ring, R.; Hennigan, T.; Huang, S.; Mag-
giore, L.; Jones, C.; Cassirer, A.; Brock, A.; Paganini,
M.;Irving, G.;Vinyals, O.;Osindero, S.;Simonyan, K.;
Rae, J.; Elsen, E.; and Sifre, L. 2022. Improving Lan-
guageModelsbyRetrievingfromTrillionsofTokens. In
Proceedingsofthe39thInternationalConferenceonMa-
chine Learning, volume 162 ofProceedings of Machine
Learning Research, 2206–2240. PMLR.
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.;
and Liu, Z. 2024. M3-Embedding: Multi-Linguality,
Multi-Functionality, Multi-Granularity Text Embed-
dingsThroughSelf-KnowledgeDistillation. InFindings
of the Association for Computational Linguistics: ACL
2024, 2318–2335. Bangkok, Thailand: Association for
Computational Linguistics.
Cheng, J.; Liu, Y.; Zhang, X.; Fei, Y.; Hong, W.; Lyu,
R.; Wang, W.; Su, Z.; Gu, X.; Liu, X.; Bai, Y.; Tang, J.;
Wang,H.;andHuang,M.2025. Glyph:ScalingContext
Windows via Visual-Text Compression.arXiv preprint
arXiv:2510.17800.
Feng, L.; Yang, F.; Chen, F.; Cheng, X.; Xu, H.; Wan,
Z.;Yan,M.;andAn,B.2026. AgentOCR:Reimagining
Agent History via Optical Self-Compression. InPro-
ceedings of the 64th Annual Meeting of the Association
forComputationalLinguistics(Volume1:LongPapers),
5067–5086. San Diego, California, United States: Asso-
ciation for Computational Linguistics.
Gim,I.;Chen,G.;Lee,S.-s.;Sarda,N.;Khandelwal,A.;
and Zhong, L. 2024. Prompt Cache: Modular Attention
Reuse for Low-Latency Inference. InProceedings of
Machine Learning and Systems, volume 6, 325–338.
GLM-V Team. 2025. GLM-4.1V-Thinking and GLM-
4.5V: Towards Versatile Multimodal Reasoning with
Scalable Reinforcement Learning.arXiv preprint
arXiv:2507.01006.
Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.-
W. 2020. Retrieval Augmented Language Model Pre-
Training. InProceedings of the 37th International Con-
ference on Machine Learning, volume 119 ofProceed-
ingsofMachineLearningResearch,3929–3938.PMLR.
Hu,J.;Huang,W.;Wang,W.;Wang,H.;Hu,T.;Qin,Z.;
Feng,H.;Chen,X.;Shan,Y.;andXie,T.2025.EPIC:Ef-
ficient Position-Independent Caching for Serving Large
Language Models. InProceedings of the 42nd Inter-
national Conference on Machine Learning, volume 267
ofProceedings of Machine Learning Research, 24391–
24402. PMLR.

Jin,C.;Zhang,Z.;Jiang,X.;Liu,F.;Liu,S.;Liu,X.;and
Jin, X. 2025. RAGCache: Efficient Knowledge Caching
forRetrieval-AugmentedGeneration.ACMTransactions
on Computer Systems, 44(1): 2:1–2:27.
Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.;
Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Pas-
sage Retrieval for Open-Domain Question Answering.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, 6769–6781.
Association for Computational Linguistics.
Kwon,W.;Li,Z.;Zhuang,S.;Sheng,Y.;Zheng,L.;Yu,
C.H.;Gonzalez,J.E.;Zhang,H.;andStoica,I.2023.Ef-
ficientMemoryManagementforLargeLanguageModel
ServingwithPagedAttention. InProceedingsofthe29th
Symposium on Operating Systems Principles, 611–626.
Koblenz,Germany:AssociationforComputingMachin-
ery.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin,
V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rock-
täschel, T.; Riedel, S.; and Kiela, D. 2020. Retrieval-
Augmented Generation for Knowledge-Intensive NLP
Tasks. InAdvances in Neural Information Processing
Systems, volume 33, 9459–9474.
Li,Y.;Lan,Z.;andZhou,J.2025. TextorPixels?Evalu-
atingEfficiencyandUnderstandingofLLMswithVisual
TextInputs. InFindingsoftheAssociationforComputa-
tionalLinguistics:EMNLP2025,10564–10578.Associ-
ation for Computational Linguistics.
Liu,Y.;Li,H.;Cheng,Y.;Ray,S.;Huang,Y.;Zhang,Q.;
Du, K.; Yao, J.; Lu, S.; Ananthanarayanan, G.; Maire,
M.; Hoffmann, H.; Holtzman, A.; and Jiang, J. 2024.
CacheGen: KV Cache Compression and Streaming for
Fast Large Language Model Serving. InProceedings of
the ACM SIGCOMM 2024 Conference, 38–56. Sydney,
NSW, Australia: Association for Computing Machinery.
Lu,S.;Wang,H.;Rong,Y.;Chen,Z.;andTang,Y.2025.
TurboRAG: Accelerating Retrieval-Augmented Genera-
tion with Precomputed KV Caches for Chunked Text.
InProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, 6588–6601.
Association for Computational Linguistics.
Qin, R.; Li, Z.; He, W.; Cui, J.; Ren, F.; Zhang, M.;
Wu,Y.;Zheng,W.;andXu,X.2025a. Mooncake:Trad-
ing More Storage for Less Computation—A KVCache-
centric Architecture for Serving LLM Chatbot. In23rd
USENIX Conference on File and Storage Technologies
(FAST25),155–170.SantaClara,CA:USENIXAssoci-
ation.
Qin,S.;Yu,H.;Wu,C.;Li,Z.;Cao,Y.;Zhuge,Z.;Zhou,
Y.;Yao,W.;Zhang,Y.;Wang,Z.;Bai,S.;Zhang,J.;and
Lin, J. 2025b. VLCache: Computing 2% Vision Tokens
and Reusing 98% for Vision-Language Inference.arXiv
preprint arXiv:2512.12977.
Su,J.;Ahmed,M.H.M.;Lu,Y.;Pan,S.;Bo,W.;andLiu,
Y.2024. RoFormer:EnhancedTransformerwithRotary
Position Embedding.Neurocomputing, 568: 127063.
Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.;
Chen, K.; Liu, X.; Wang, J.; Ge, W.; Fan, Y.; Dang, K.;
Du, M.; Ren, X.; Men, R.; Liu, D.; Zhou, C.; Zhou,
J.; and Lin, J. 2024. Qwen2-VL: Enhancing Vision-
Language Model’s Perception of the World at Any Res-
olution.arXiv preprint arXiv:2409.12191.Wang, Z.; Ma, D.; Zhong, H.; Li, J.; Zhang, W.; Wang,
B.; and He, C. 2026. AgenticOCR: Parsing Only What
YouNeedforEfficientRetrieval-AugmentedGeneration.
arXiv preprint arXiv:2602.24134.
Wei, H.; Sun, Y.; and Li, Y. 2025. DeepSeek-
OCR: Contexts Optical Compression.arXiv preprint
arXiv:2510.18234.
Wei, H.; Sun, Y.; and Li, Y. 2026. DeepSeek-OCR 2:
Visual Causal Flow.arXiv preprint arXiv:2601.20552.
Xing,L.;Wang,A.J.;Yan,R.;Shu,X.;andTang,J.2025.
Vision-Centric Token Compression in Large Language
Model. InAdvances in Neural Information Processing
Systems, volume 38, 37239–37269.
Yao,J.;Li,H.;Liu,Y.;Ray,S.;Cheng,Y.;Zhang,Q.;Du,
K.; Lu, S.; and Jiang, J. 2025. CacheBlend: Fast Large
Language Model Serving for RAG with Cached Knowl-
edge Fusion. InProceedings of the Twentieth European
Conference on Computer Systems, 94–109. Association
for Computing Machinery.
Zhao, H.; Wang, M.; Zhu, F.; Liu, W.; Ni, B.; Zeng, F.;
Meng,G.;andZhang,Z.2025a.VTCBench:CanVision-
LanguageModelsUnderstandLongContextwithVision-
Text Compression?arXiv preprint arXiv:2512.15649.
Zhao, S.; Hu, J.; Huang, R.; Zheng, J.; and Chen, G.
2025b. MPIC: Position-Independent Multimodal Con-
textCachingSystemforEfficientMLLMServing.arXiv
preprint arXiv:2502.01960.
Zheng,L.;Yin,L.;Xie,Z.;Sun,C.;Huang,J.;Yu,C.H.;
Cao,S.;Kozyrakis,C.;Stoica,I.;Gonzalez,J.E.;Barrett,
C.; and Sheng, Y. 2024. SGLang: Efficient Execution
of Structured Language Model Programs. InAdvances
in Neural Information Processing Systems, volume 37,
62557–62583.