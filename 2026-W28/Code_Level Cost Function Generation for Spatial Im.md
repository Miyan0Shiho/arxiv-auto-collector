# Code-Level Cost Function Generation for Spatial Image Steganography Using RAG-Enhanced Large Language Models

**Authors**: Yige Wang, Shiqi Yi, Hanzhou Wu

**Published**: 2026-07-07 05:58:33

**PDF URL**: [https://arxiv.org/pdf/2607.05868v1](https://arxiv.org/pdf/2607.05868v1)

## Abstract
Designing cost functions of adaptive steganography traditionally requires extensive manual tuning, while deep learning methods lack interpretability. Although large language models (LLMs) offer an automated alternative via evolutionary generation, they often violate domain specific mathematical constraints due to a lack of explicit domain knowledge. To address this problem, we propose a novel evolutionary system focused on exploiting Retrieval-Augmented Generation (RAG) enhanced LLMs for the automatic code-level generation of spatial steganography cost functions. This system incorporates a core Self Evolving RAG (SE-RAG) module, wherein a Code Semantic Signature (CSS) translates procedural code into aligned queries, retrieving explicit guidance from static literature and dynamic experience knowledge bases to steer the LLM generation process. A dedicated feedback mechanism then continuously refines the dynamic knowledge base with successful optimization strategies. Extensive experiments on the BOSSBase and BOWS2 datasets demonstrate that the proposed framework consistently achieves higher steganographic security than existing automatically designed methods, and increases the average code execution rate by 46.3% while reducing the search cost by 26.1%, thereby highlighting the effectiveness, efficiency, and potential of combining LLMs with domain-specific knowledge in the field of automatic steganographic algorithm generation.

## Full Text


<!-- PDF content starts -->

Code-Level Cost Function Generation for Spatial
Image Steganography Using RAG-Enhanced Large
Language Models
Yige Wang
School of Communication & Information
Engineering, Shanghai University
Shanghai 200444, China
24721152@shu.edu.cnShiqi Yi
School of Communication & Information
Engineering, Shanghai University
Shanghai 200444, China
shiqiyi2025@163.comHanzhou Wu∗
School of Communication & Information
Engineering, Shanghai University
Shanghai 200444, China
h.wu.phd@ieee.org
Abstract—Designing cost functions of adaptive steganogra-
phy traditionally requires extensive manual tuning, while deep
learning methods lack interpretability. Although large language
models (LLMs) offer an automated alternative via evolutionary
generation, they often violate domain specific mathematical
constraints due to a lack of explicit domain knowledge. To
address this problem, we propose a novel evolutionary system
focused on exploiting Retrieval-Augmented Generation (RAG)
enhanced LLMs for the automatic code-level generation of spatial
steganography cost functions. This system incorporates a core
Self Evolving RAG (SE-RAG) module, wherein a Code Semantic
Signature (CSS) translates procedural code into aligned queries,
retrieving explicit guidance from static literature and dynamic
experience knowledge bases to steer the LLM generation process.
A dedicated feedback mechanism then continuously refines the
dynamic knowledge base with successful optimization strategies.
Extensive experiments on the BOSSBase and BOWS2 datasets
demonstrate that the proposed framework consistently achieves
higher steganographic security than existing automatically de-
signed methods, and increases the average code execution rate
by 46.3% while reducing the search cost by 26.1%, thereby high-
lighting the effectiveness, efficiency, and potential of combining
LLMs with domain-specific knowledge in the field of automatic
steganographic algorithm generation.
Index Terms—Adaptive steganography, cost function, security,
large language models, retrieval augmented generation.
I. INTRODUCTION
Image steganography conceals secret information within in-
nocuous cover images while minimizing detectable embedding
artifacts. Among existing adaptive steganographic approaches,
the minimum-distortion framework has achieved remarkable
success due to its flexibility and effectiveness, where embed-
ding costs are typically manually assigned according to local
image characteristics and optimized using syndrome trellis
codes (STCs) [1]. Although highly effective, designing distor-
tion cost functions still relies heavily on expert knowledge and
extensive trial and error [2]–[4]. Recent deep learning based
approaches alleviate manual feature engineering by learning
steganographic representations directly from data, at the cost
∗Author to whom any correspondence should be addressed.of large training data requirements and limited interpretabil-
ity. Recently, large language models (LLMs) have begun to
be applied to steganographic tasks [6]. They have further
emerged as a promising alternative for automatic algorithm
design through evolutionary code generation [5]. However,
without explicit domain knowledge, language models tend to
perform unguided evolutionary modifications that may violate
mathematical constraints or deviate from effective distortion
optimization.
Unlike general code generation, automatic steganographic
cost function design is fundamentally a knowledge-intensive
task. The objective is not merely to generate executable
programs, but to generate mathematically valid distortion func-
tions that satisfy strict steganographic constraints while im-
proving resistance against steganalysis. Such domain-specific
knowledge cannot be fully captured by the general-purpose
knowledge embedded in LLMs alone. Updating the internal
knowledge of LLMs through continual training [7] or model
editing [8] is computationally expensive and difficult to adapt
to the rapidly evolving search process. Retrieval-augmented
generation (RAG) [9], [10] has therefore emerged as an effec-
tive paradigm for augmenting LLMs with external knowledge
without modifying the model parameters. Recent studies have
consequently extended RAG through iterative retrieval [11],
self-reflective retrieval [12], [13], and evolutionary retrieval
strategies [14]. Meanwhile, retrieval-enhanced code generation
has also demonstrated the benefit of exploiting structured ex-
ternal knowledge for program synthesis [15]. These advances
demonstrate that retrieval provides an effective mechanism
for incorporating task-specific knowledge into LLM-based
generation, making it a promising foundation for automatic
steganographic algorithm design.
Motivated by these observations, we propose a knowledge-
guided evolutionary framework, which incorporates a self-
evolving RAG (SE-RAG) module to provide structured domain
guidance throughout the evolutionary process. Specifically, the
SE-RAG module jointly exploits a static literature knowledge
base and a dynamic experience knowledge base to achieve this
goal. We further introduce a code semantic signature (CSS)
arXiv:2607.05868v1  [cs.CR]  7 Jul 2026

Fig. 1. Overview of the proposed framework.
that bridges procedural implementations and steganographic
concepts, enabling semantics-aware retrieval instead of syntax-
oriented matching. A feedback-driven knowledge evolution
mechanism continuously transforms successful optimization
strategies into reusable experience, allowing subsequent gen-
erations to evolve under progressively richer domain guidance.
The main contributions are summarized as follows:
•We propose a knowledge-driven evolutionary framework
for automatic steganographic cost function design, em-
powered by a novel SE-RAG module that integrates a
static literature knowledge base with a dynamic experi-
ence knowledge base.
•We introduce CSS that bridges the semantic gap between
program implementations and steganographic concepts,
improving retrieval relevance for cost function generation.
•We develop a novel feedback-driven knowledge evolution
mechanism that continuously archives successful opti-
mization experiences into the dynamic knowledge base
to provide progressively richer domain guidance.
•Comprehensive evaluations demonstrate that the algo-
rithms generated by our framework consistently achieve
higher steganographic security than existing paradigms.
Furthermore, the framework significantly enhances evolu-
tionary efficiency, increasing the average code execution
rate by 46.3% and reducing the search cost by 26.1%.
II. PROPOSEDFRAMEWORK
A. Overall Architecture
To advance automated cost function design within the
minimum-distortion framework, we establish a knowledge-
driven evolutionary paradigm. While recent automated frame-
works rely on unguided two-stage generation [5], our architec-
ture fundamentally transforms this process by establishing the
SE-RAG module as the core engine for explicit domain guid-
ance. The framework evolves over multiple generations, each
comprising cost function generation and evaluation stages.
As illustrated in Fig. 1, the process is initialized with a
Fig. 2. The internal pipeline of the SE-RAG module.
foundational handcrafted steganographic algorithmp init, which
acts as the initial program.
Stage 1 focuses on algorithm generation and preliminary
evaluation. In each iteration, a reference programp refis
selected from the program database and fed into the forward
process of SE-RAG. Crucially, the CSS ofp refis extracted to
perform separate dual-path retrievals from a static literature
knowledge baseK staand a dynamic experience knowledge
baseKt
dyn, supplying the LLM with classic academic concepts
and historically successful optimization paths. Guided by this
retrieved context, the LLM optimizesp refto generate new
candidate algorithms (p 1, p2, . . . , p n). These variants undergo
rapid preliminary evaluation via pretrained steganalyzers to
filter out non-executable code. We uniformly utilize the min-
imum average decision error rateP Eas the security metric:
PE= min
PFA1
2(PFA+P MD(PFA)),(1)
whereP FAandP MDrepresent the false alarm and missed
detection rates, respectively. Valid programs and their prelim-
inaryP Escores are saved back into the program database.
Stage 2 conducts precise evaluation for candidates passing
the preliminary filter to provide feedback. A dedicated ste-
ganalyzer is trained for each program to provide a rigorous
assessment via theP Emetric defined in Eq. (1). The results of
this accurate evaluation update the pretrained steganalyzer pool
in Stage 1, and the optimized programs are added to the seed
program pool for subsequent generations. Simultaneously, the
evaluation results leverage the backward process of SE-RAG to
update the dynamic experience knowledge baseKt
dynbased on
the evaluation results, accumulating effective empirical paths
for continuous improvement.
The overall workflow is illustrated in Fig. 1, and the internal
pipeline of the SE-RAG module is detailed in Fig. 2. Section
II-B describes the forward generation process, while Section
II-C outlines the backward feedback loop.
B. SE-RAG: Forward Generation Process
The forward generation process transitions the LLM from a
blind code generator into a knowledge-informed algorithm op-

TABLE I
MAPPING FROMPYTHONCODEPATTERNS TOSEMANTICCONCEPTS
Raw Python Operation Semantic Concept
convolve+reciprocalTexture Detection
convolve after reciprocalSpatial Cost Smoothing
rollEdge Offset
sliceSpatial Cropping
>= 255/<= 0Boundary Value Constraints
1 / xCost Initialization Inverse
timizer via domain knowledge injection. To achieve effective
retrieval, relying directly on the raw reference programp ref
as a query is suboptimal due to the conceptual gap between
procedural implementations and steganographic semantics. To
bridge this gap, as illustrated in Fig. 2, we introduce the CSS
mechanism, wherep refis parsed by an AST parser. This parser
maps specific Python operations to high-level steganographic
semantic concepts based on the mapping rules standardized in
Table I, sequentially connecting them into an ordered semantic
sequenceS(p).
Prior to retrieval, literature documents and historically suc-
cessful rules are partitioned into semantic text chunksk. These
chunks populate two repositories, namely, the Static Knowl-
edge BaseK staindexing fundamental reference papers, and the
Dynamic Knowledge BaseKt
dyn, which continuously expands
at each evolutionary iterationtto store concise optimization
summaries. Formally, each repositoryK i∈ {K sta,Kt
dyn}is de-
fined as a set of constituent chunks:K i={k 1, k2, . . . , k |Ki|}.
All chunks are processed by a Bi-Encoder embedding model
MBEand stored alongside their continuous vector representa-
tions.
During the retrieval phase, the obtained semantic signature
formulates the queryq=S(p ref). A parallel two-stage retrieve-
and-rerank pipeline is independently applied to each reposi-
tory. First,M BEprojectsqinto the latent space for a rapid
vector screening, whereqis compared against every chunk
k∈ K ito calculate the preliminary similarity scores 1(q, k)
via cosine similarity:
s1(q, k) =MBE(q)·M BE(k)
∥M BE(q)∥∥M BE(k)∥.(2)
This screens the search space to retain a highly relevant
matching poolM i,q. Second, the matching pools are for-
warded to a Cross-Encoder modelM CEfor precise reranking
to capture fine-grained cross-attention. The final ranking score
is computed as:
s2(q, k) =M CE(q, k),∀k∈ M i,q.(3)
Finally, a prompt compiler systematically aggregates the ac-
quired information into a highly structured prompt to govern
the LLM code synthesis. Instead of a simple concatenation, the
compilation process utilizes a sequential integration function
Φto combine four essential components: First, the expert task
guidelinesT, which assign a domain expert persona to the
language model and explicitly mandate structural algorithmic
innovations, such as novel residual sources or nonadditive fu-sion rules, while strictly prohibiting superficial parameter tun-
ing; Second, the retrieved contextC q, which supplies explicit
domain guidance and historical optimization strategies; Third,
the reference programp refundergoing evolution; and Fourth,
the rigorous safety constraintsR, which enforce deterministic
execution, restrict external dependencies to standard numer-
ical packages, and define rigorous mathematical boundaries
to satisfy necessary steganographic security constraints. This
compilation operation is formally defined as:
P= Φ(T,C q, pref,R).(4)
The resulting knowledge-injected prompt is then delivered
to the LLM to generate the new batch of program candidates.
C. SE-RAG: Backward Feedback Loop
To prevent the evolutionary process from stalling or repeat-
ing past optimization errors, the proposed framework incorpo-
rates a backward semantic knowledge extraction mechanism
driven by a semantic auditor. The core purpose of this com-
ponent is to transform fortuitous performance improvements
into interpretable steganographic rules, which are subsequently
leveraged to guide future mutation directions.
As illustrated in the backward process of Fig. 2, during
the accurate evaluation phase, a dedicated steganalysis tool
evaluates the optimized programp cto obtain its security score
PE(pc), as defined in Eq. (1). If this score exceeds that of the
initial global baseline, i.e.,P E(pc)> P E(pinit), the knowledge
extraction mechanism inside the Semantic Auditor is triggered.
The framework employs an independent LLM summarizer
within the auditor to perform a detailed comparative analysis
betweenp candp init. By examining the algorithmic structures
and semantic variations, the summarizer extracts and summa-
rizes the specific methodological modifications that lead to the
enhanced security. The empirical insights generated during the
current iterationtform a set:
Et
new={e(1), e(2), . . .}.(5)
Consistent with the knowledge representation established in
Section II-B, each empirical experiencee∈Et
newis partitioned
into semantic text chunks, embedded into vector represen-
tations, and structurally archived. LetKt
new={k 1, k2, . . .}
denote the set of all semantic chunks derived fromEt
new. The
dynamic knowledge base is updated via a direct set union
operation:
Kt
dyn=Kt−1
dyn∪ Kt
new,(6)
whereK0
dyn=∅.
Integrating this structured knowledge back into the loop es-
tablishes a sustainable self-evolving mechanism. As evolution
progresses, the dynamic knowledge baseKt
dyncontinuously
accumulates historically successful design rules. Consequently,
when the next iteration performs forward retrieval, the retrieval
engine draws directional guidance from both the foundational
static knowledge baseK staand the updated dynamic knowl-
edge baseKt
dyn, drastically accelerating convergence toward
highly secure steganographic cost functions.

TABLE II
DETECTION ERROR OF ORIGINAL ALGORITHMS AND THE BEST EVOLVED
ALGORITHMS DISCOVERED BY DIFFERENT AUTOMATIC DESIGN
FRAMEWORKS UNDER DIFFERENT EMBEDDING PAYLOADS ONBOSSBASE
(HIGHER VALUES INDICATE STRONGER SECURITY).
Algorithm FrameworkEmbedding Payload (bpp)
0.4 0.3 0.2 0.1
HILLOriginal 0.1574 0.2063 0.2596 0.3449
Ref. [5] 0.1654 0.2241 0.2700 0.3559
Ours0.2036 0.2433 0.3064 0.3878
WOWOriginal 0.1053 0.1361 0.1800 0.2640
Ref. [5] 0.1385 0.1695 0.2180 0.2911
Ours0.1419 0.1711 0.2249 0.3186
S-UNIWARDOriginal 0.1139 0.1573 0.2169 0.3245
Ref. [5] 0.1245 0.1709 0.2309 0.3376
Ours0.1588 0.2033 0.2588 0.3608
III. EXPERIMENTALSETUP
Evolutionary generation and evaluation are conducted on
BOSSBase v1.01 [19], with optimal algorithms further tested
on BOWS2 [20] for generalization. Both datasets contain
10,000256×256grayscale images, strictly partitioned at
a 5:1:4 ratio for paired cover-stego training, validation, and
testing. All algorithms are implemented in Python via an
embedding simulator.
OpenAI GPT-4o [16] serves as the core LLM for both code
generation and backward experience summarization. Knowl-
edge bases are managed via Chroma DB using chunks of 1,000
characters with a stride of 800. We employ the Bi-Encoder
all-MiniLM-L6-v2[18] for dense retrieval and the Cross-
Encoderms-marco-MiniLM-L-6-v2for reranking. The
static repository indexes 228 classical steganography papers.
To balance noise and length disparities, the retrieve-and-rerank
pipeline filters the top 50 static and 20 dynamic chunks down
toN sta= 3andN dyn= 5, respectively. The evolution is
initialized with WOW [3], HILL [2], and S-UNIWARD [4].
Adopting hyperparameters from [5], each prompt directs the
LLM to synthesize four candidate variants per iteration.
For search efficiency, evolution executes exclusively at the
highest payload of 0.4 bpp using a fully trained SRNet
[17]. The discovered algorithms are then evaluated across
lower payloads from 0.3 to 0.1 bpp. The SRNet training
parameters and progressive fine-tuning strategy strictly follow
the configurations in [5] and the original implementation [17].
IV. RESULTS ANDANALYSIS
A. Steganographic Security
To evaluate the effectiveness of the proposed framework,
we compare the steganographic security of the original hand-
crafted algorithms and the best algorithms generated by our
framework with those produced by the existing LLM-based
approach [5]. HILL [2], WOW [3], and S-UNIWARD [4] are
adopted as the starting algorithms for the evolutionary search
process in both frameworks. Security is quantified by the
detection error rateP Edefined in Eq. (1), which is evaluated
using dedicated SRNet [17] steganalyzers trained specifically
for each evaluated algorithm at each embedding payload.TABLE III
DETECTION ERROR OF ORIGINAL ALGORITHMS AND THE BEST EVOLVED
ALGORITHMS DISCOVERED BY DIFFERENT AUTOMATIC DESIGN
FRAMEWORKS UNDER DIFFERENT EMBEDDING PAYLOADS ONBOWS2
(HIGHER VALUES INDICATE STRONGER SECURITY).
Algorithm FrameworkEmbedding Payload (bpp)
0.4 0.3 0.2 0.1
HILLOriginal 0.2515 0.2978 0.3601 0.4279
Ref. [5] 0.2738 0.3198 0.3739 0.4450
Ours0.3476 0.4101 0.4584 0.4848
WOWOriginal 0.1605 0.1974 0.2680 0.3644
Ref. [5] 0.2189 0.2604 0.3284 0.4093
Ours0.2344 0.2811 0.3515 0.4451
S-UNIWARDOriginal 0.1931 0.2536 0.3174 0.4229
Ref. [5] 0.2194 0.2679 0.3499 0.4378
Ours0.2541 0.2999 0.3620 0.4510
TABLE IV
EVOLUTION EFFICIENCY COMPARISON BETWEEN DIFFERENT AUTOMATIC
DESIGN FRAMEWORKS.
Metric FrameworkInitial Algorithm
HILL WOW S-UNIWARD
Execution Rate (↑)Ref. [5] 35.0% 36.3% 20.0%
Ours57.5% 54.2% 21.9%
Target Iterations (↓)Ref. [5] 100 160 240
Ours80 120 160
As reported in Table II and Table III, while the algorithms
produced by both automated paradigms successfully enhance
the security of their handcrafted counterparts, those gener-
ated by our framework consistently achieve the highestP E
across varying payloads on both BOSSBase and BOWS2.
By continuously outperforming the baseline across divergent
steganographic paradigms, this framework-level comparison
systematically demonstrates the superior optimization capacity
of our evolutionary framework incorporating the SE-RAG en-
gine, proving that the security gains stem from methodological
advancements rather than stochastic fluctuations during the
evolutionary search. Consequently, rather than relying solely
on the implicit knowledge of the language model, our frame-
work explicitly incorporates domain-specific knowledge into
the evolutionary process via the SE-RAG module, enabling a
more meaningful exploration of the program space and con-
sistently producing higher-quality steganographic algorithms.
B. Evolution Efficiency Analysis
To further evaluate the proposed framework, we analyze
its evolutionary efficiency in terms of code executability,
defined as the percentage of successfully executed candidate
algorithms, and search efficiency, representing the number of
generated algorithms required to discover the best-performing
solution.
As shown in Table IV, the proposed framework improves
the overall code execution rate across all three initial al-
gorithms, increasing the average execution rate from 30.4%
to 44.5%, which corresponds to a relative improvement of
46.3%. Although the execution rate enhancement for the

highly constrained S-UNIWARD is marginal, the overall
search efficiency is consistently optimized. Specifically, our
framework requires substantially fewer generated algorithms
to discover the best-performing solution across all baselines,
reducing the overall search cost by an average of 26.1%.
Collectively, despite the inherent challenges in optimizing
complex spatial steganographic models like S-UNIWARD,
these results demonstrate that our framework enables more
reliable candidate generation and highly efficient optimization.
This substantially reduces the computational cost of automatic
steganographic algorithm design, highlighting its practical
applicability.
C. Analysis of Evolved Cost Functions
To explore the effectiveness of our evolutionary system, we
present the evolved cost functions derived from HILL, WOW,
and S-UNIWARD, detailing the specific modifications made
to each algorithm.
1) Evolved HILL:The original HILL algorithm estimates
distortion via a high-pass residualR H=H 1⊗C, whereC∈
RM×Nis the cover image,⊗denotes convolution, andH 1is
the3×3Ker-Bohme (KB) filter. It formulates the suitability
map and cost as:
ξorig=|R H| ⊗L 1, ρorig=1
ξorig+ϵ⊗L 2,(7)
whereL 1andL 2are3×3and15×15mean filters,
respectively, andϵ= 10−8ensures numerical stability.
The evolved algorithm refines this by introducing a median
residualR M=C−Med 3×3(C), where Med 3×3(·)denotes a
3×3local median filter. These residuals are fused into a joint
response:
RF=p
|RH·RM|,(8)
where·,| · |, and√·represent element-wise multiplication,
absolute value, and square root. To measure texture complex-
ity, the local varianceV(C) i,j=Var(Ω 3×3(Ci,j))within a
3×3neighborhoodΩ 3×3 at each coordinate(i, j)is com-
puted, and immediately truncated into a variance gating factor
˜V(C) i,j=clip(V(C) i,j, ϵ,5), where clip(x, a, b)restrictsx
within[a, b]. Finally, the evolved algorithm constructs a refined
suitability mapξand maps it directly to the local embedding
costρ i,j:
ξ=R F⊗L 1, ρi,j=1
(ξi,j+ϵ)· ˜V(C) i,j.(9)
To enforce physical boundaries and numerical stability, valid
costs are lower-bounded byϵ. Extreme numerical values and
invalid boundary modifications are strictly penalized with an
infinite wet cost.
2) Evolved WOW:The original WOW employs Daubechies
8-tap (DB-8) low-pass (h) and high-pass (g) filters to construct
three directional filters via outer products to represent the LH,
HL, and HH subbands:K(0)=hg⊤,K(1)=gh⊤, and
K(2)=gg⊤. The embedding suitability maps quantify local
texture complexity by measuring weighted absolute residual
differences:
ξ(k)
orig=|C⊗K(k)| ⊙ |K(k)|, k∈ {0,1,2},(10)where⊙represents mirror-padded correlation. Pixels with
small directional residuals are highly predictable, thus receiv-
ing higher embedding costs to minimize detectability. The
original cost aggregates the reciprocal of these maps:
ρorig=1
ξ(0)
orig+1
ξ(1)
orig+1
ξ(2)
orig.(11)
Replacing conventional hand-crafted wavelets, our evolu-
tionary system autonomously discovers an asymmetric 9-tap
filter pair, denoted ash eandg e. Substituting the DB-8 vectors,
it reconstructs the three standard subbands and introduces a
fourth low-frequency structural subbandK(3)=h eh⊤
eto
capture local structural consistency. The expanded suitability
maps use the identical formulation:
ξ(m)=|C⊗K(m)| ⊙ |K(m)|, m∈ {0,1,2,3}.(12)
The evolved algorithm replaces static reciprocal aggrega-
tion with a pixel-wise dynamic fusion mechanism. Specifi-
cally, the directional discrepancy is first evaluated using the
channel-wise varianceV(ξ) i,j=Var
ξ(0)
i,j, ξ(1)
i,j, ξ(2)
i,j, ξ(3)
i,j
,
which is scaled via global normalization to yield ˜Vξ(i, j) =
V(ξ) i,j/(max(V(ξ)) +ϵ). A dynamic pooling parameterP i,j
is then generated to locally adjust the fusion intensity:
Pi,j=−0.7 + 0.3 ˜Vξ(i, j).(13)
To strictly ensure mathematical validity across spatially
varying parameters, the aggregation is performed at each pixel
by defining a joint suitability poolΨ i,j:
Ψi,j=3X
m=0
ξ(m)
i,jPi,j
, ρi,j= Ψ−1/P i,j
i,j .(14)
Cost bounding and numerical stabilization follow the iden-
tical protocols established in the evolved HILL algorithm.
3) Evolved S-UNIWARD:The original S-UNIWARD eval-
uates distortion using directional relative change profiles. Fol-
lowing WOW, three DB-8 directional filters, denoted byK(k)
fork∈ {0,1,2}, are employed to compute the directional
suitability maps:
ξ(k)
orig=1
|C⊗K(k)|+σ⊙ |K(k)|, k∈ {0,1,2},(15)
whereσ= 1is a stabilization constant. The overall embedding
cost is then obtained by aggregating the three directional
suitability maps:
ρorig=ξ(0)
orig+ξ(1)
orig+ξ(2)
orig.(16)
The evolved algorithm expands this design by augmenting
the filter bank. Retaining the classic DB-8 filters, it introduces
an approximate horizontal gradient filterK(3)= [−0.5,0,0.5]
and its vertical counterpartK(4)= (K(3))⊤to evaluate local
boundary discontinuities. To ensure optimal compliance with
the physical properties of different residuals, the expanded
suitability maps are extracted through divergent smoothing
strategies. The gradient subbands employ a5×5local me-

dian filter, while the first three subbands retain their original
strategies:
ξ(m)=

1
|C⊗K(m)|+σ⊙ |K(m)|, m∈ {0,1,2},
Med 5×5
1
|C⊗K(m)|+σ
, m∈ {3,4}.(17)
A geometric mean aggregation supersedes the original linear
additive summation in this expanded design. The final embed-
ding cost is therefore formulated by explicitly evaluating the
joint product of all five suitability tracks at each pixel:
ρi,j=
ξ(0)
i,j·ξ(1)
i,j·ξ(2)
i,j·ξ(3)
i,j·ξ(4)
i,j1/5
.(18)
As with the preceding algorithms, physical boundaries and
extreme numerical values are regulated through identical con-
straint mechanisms.
V. CONCLUSION
In this paper, we propose a knowledge-based evolutionary
framework for the generation of code-level cost functions
in spatial image steganography. To overcome the lack of
domain-specific guidance in existing LLM-based automated
design, our framework is driven by a central SE-RAG module.
Within this engine, CSS is utilized to bridge the semantic
gap between procedural implementations and steganographic
principles, enabling the continuous and dynamic integration
of explicit guidance from both static literature and historically
successful optimization experiences.
Experimental results on BOSSBase and BOWS2 demon-
strate that our framework consistently achieves higher stegano-
graphic security while improving code executability and evolu-
tionary efficiency over existing methods. Crucially, by explic-
itly injecting structured domain knowledge into the generation
loop, this architecture successfully transforms the LLM from a
blind code synthesizer into a directed algorithm optimizer. This
knowledge-driven guidance effectively prevents mathemati-
cally invalid mutations and accelerates convergence toward
highly secure cost functions. Future research will focus on
expanding the analytical capabilities of the semantic auditor,
broadening the evaluation framework across a wider spectrum
of steganalyzers to validate universal algorithmic resistance,
and adapting this offline knowledge-driven architecture to
broader media forensics applications.
ACKNOWLEDGMENTS
This research was supported by the Science and Technology
Commission of Shanghai Municipality (STCSM) under grant
24ZR1424000, National Natural Science Foundation of China
(NSFC) under grant U23B2023, and the Nanning “Yong Jiang”
Program under grant RC20250102.
REFERENCES
[1] T. Filler, J. Judas, and J. Fridrich, “Minimizing additive distortion in
steganography using syndrome-trellis codes,”IEEE Transactions on
Information Forensics and Security, vol. 6, no. 3, pp. 920–935, Sep.
2011.
[2] B. Li, M. Wang, J. Huang, and X. Li, “A new cost function for
spatial image steganography,” inProceedings of the IEEE International
Conference on Image Processing, pp. 4206–4210, Oct. 2014.[3] V . Holub and J. Fridrich, “Designing steganographic distortion using
directional filters,” inProceedings of the IEEE International Workshop
on Information Forensics and Security, pp. 234–239, Dec. 2012.
[4] V . Holub and J. Fridrich, “Digital image steganography using universal
distortion,” inProceedings of the 1st ACM Workshop on Information
Hiding and Multimedia Security, pp. 1–4, Jun. 2013.
[5] H. Wu and Y . Wang, “Defining cost function of steganography with
large language models,” inProceedings of the IS&T Electronic Imaging,
Media Watermarking, Security, and Forensics, pp. 307-1–307-14, 2026.
[6] H. Wu, “Prompting steganography: a new paradigm,” inProceedings
of the IS&T Electronic Imaging, Media Watermarking, Security, and
Forensics, pp. 338-1–338-11, 2024.
[7] Z. Ke, H. Lin, Y . Shao, H. Xu, L. Shu, and B. Liu, “Continual training
of language models for few shot learning,” inProceedings of the
Conference on Empirical Methods in Natural Language Processing, pp.
10205–10216, 2022.
[8] K. Meng, D. Bau, A. J. Andonian, and Y . Belinkov, “Locating and
editing factual associations in GPT,” inAdvances in Neural Information
Processing Systems, vol. 35, pp. 17359–17372, 2022.
[9] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdvances in Neural Information Processing Systems,
vol. 33, pp. 9459–9474, 2020.
[10] P. Zhaoet al., “Retrieval-augmented generation for AI-generated con-
tent: A survey,”Data Science and Engineering, vol. 11, no. 1, pp. 1–29,
Jan. 2026.
[11] Z. Shao, Y . Gong, Y . Shen, M. Huang, N. Duan, and W. Chen, “Enhanc-
ing retrieval-augmented large language models with iterative retrieval-
generation synergy,” inFindings of the Association for Computational
Linguistics: EMNLP, pp. 9248–9274, Dec. 2023.
[12] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning
to retrieve, generate, and critique through self-reflection,” inProceedings
of the 12th International Conference on Learning Representations, 2024.
[13] T. Chinthala, “Bidirectional RAG: Safe self-improving retrieval-
augmented generation through multi-stage validation,”arXiv preprint
arXiv:2512.22199, 2025.
[14] H. Suet al., “EvoR: Evolving retrieval for code generation,” inFindings
of the Association for Computational Linguistics: EMNLP, pp. 2538–
2554, Nov. 2024.
[15] F. Zhanget al., “RepoCoder: Repository level code completion through
iterative retrieval and generation,” inProceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing, pp. 2471–2484,
2023.
[16] A. Hurstet al., “GPT-4o system card,”arXiv preprint arXiv:2410.21276,
2024.
[17] M. Boroumand, M. Chen, and J. Fridrich, “Deep residual network
for steganalysis of digital images,”IEEE Transactions on Information
Forensics and Security, vol. 14, no. 5, pp. 1181–1193, 2018.
[18] W. Wang, F. Wei, L. Dong, H. Bao, M. Yang, and M. Zhou, “MiniLM:
Deep self-attention distillation for task-agnostic compression of pre-
trained transformers,” inAdvances in Neural Information Processing
Systems, vol. 33, pp. 5776–5788, 2020.
[19] P. Bas, T. Filler, and T. Pevný, “Break our steganographic system: The
ins and outs of organizing BOSS,” inProceedings of the International
Workshop on Information Hiding, pp. 59–70, May 2011.
[20] P. Bas and T. Furon, “Break our watermarking system 2,” Jul. 2007.
[Online]. Available: http://bows2.ec-lille.fr