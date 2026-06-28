# Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale

**Authors**: Yanhua Cheng, Bo Wang, Haotian Zhang, Xinyuan Gao, Zhihui Yin, Ben Xue, Yongzhi Li, Jieting Xue, Ye Ma, Minquan Wang, Jiahui Li, Tianyu Xu, Zhiqiang Liu, Xiao Lin, Shiyang Wen, Changcheng Li, Liu Liu, Quan Chen, Peng Jiang, Kun Gai

**Published**: 2026-06-24 07:23:20

**PDF URL**: [https://arxiv.org/pdf/2606.25496v1](https://arxiv.org/pdf/2606.25496v1)

## Abstract
Traditional short-video recommendation systems match user interest to a fixed pool of pre-produced videos, which limits their ability to capture fine-grained and dynamic preferences. We propose Recommendation-as-Generation (RaG), a new paradigm that generates personalized videos on demand from inferred user interest. Our framework unifies generative recommendation and video generation through shared semantic IDs (SIDs), which disentangle video representation into content semantics and creative style semantics, enabling both fine-grained modeling of user interest and controllable generation of interest-aligned videos. We further develop Video Generation Agents (VGAs) that are conditioned on inferred SIDs to drive hierarchical planning and refinement for video creation, including visual composition, audio alignment, and artistic effect enhancement. To optimize the framework, we effectively introduce a synergistic cross-domain reward learning mechanism that jointly enforces interest alignment, user feedback, and video quality assessment.
  We deploy RaG on an industrial-scale platform with over 400 million daily active users and evaluate it in a revenue-critical advertising scenario. Online A/B tests show up to 1.87% ad revenue improvement compared to a strong production GRM baseline, demonstrating its effectiveness in driving further revenue gains beyond generative recommendation. Our results highlight a closed-loop generative system as a promising paradigm for integrating personalized video generation into recommendation.

## Full Text


<!-- PDF content starts -->

Recommendation as Generation: Unifying Personalized Video
Generation and Recommendation at Industrial Scale
Yanhua Cheng∗1, Bo Wang∗1, Haotian Zhang∗§2, Xinyuan Gao∗1, Zhihui Yin1, Ben Xue1, Yongzhi Li1,
Jieting Xue1, Ye Ma1, Minquan Wang1, Jiahui Li1, Tianyu Xu1, Zhiqiang Liu1, Xiao Lin1,
Shiyang Wen1, Changcheng Li1, Liu Liu2, Quan Chen1, Peng Jiang†1, Kun Gai1
1Kuaishou Technology
Beijing, China2Beihang University
Beijing, China
Abstract
Traditional short-video recommendation systems match user in-
terest to a fixed pool of pre-produced videos, which limits their
ability to capture fine-grained and dynamic preferences. We propose
Recommendation-as-Generation(RaG), a new paradigm that
generates personalized videos on demand from inferred user inter-
est. Our framework unifies generative recommendation and video
generation through shared semantic IDs (SIDs), which disentan-
gle video representation into content semantics and creative style
semantics, enabling both fine-grained modeling of user interest
and controllable generation of interest-aligned videos. We further
developVideo Generation Agents(VGAs) that are conditioned
on inferred SIDs to drive hierarchical planning and refinement
for video creation, including visual composition, audio alignment,
and artistic effect enhancement. To optimize the framework, we
effectively introduce a synergistic cross-domain reward learning
mechanism that jointly enforces interest alignment, user feedback,
and video quality assessment.
We deploy RaG1on an industrial-scale platform with over 400
million daily active users and evaluate it in a revenue-critical adver-
tising scenario. Online A/B tests show up to1.87%ad revenue im-
provement compared to a strong production GRM baseline, demon-
strating its effectiveness in driving further revenue gains beyond
generative recommendation. Our results highlight a closed-loop
generative system as a promising paradigm for integrating person-
alized video generation into recommendation.
Keywords
Generative Recommendation, Personalized Video Generation,
Agents, Semantic Quantization, Reward Learning
1 Introduction
Over the past decade, industrial video recommendation systems
have followed a content-first paradigm, where videos are pro-
duced offline and recommendation models retrieve and rank items
from a fixed pool. Deep learning recommendation models (DL-
RMs) [ 3,5,34,36] improve matching accuracy under this setting.
More recently, generative recommendation models (GRMs) [ 7,27]
extend this paradigm by modeling user interest through large-scale
autoregressive generation over semantic IDs (SIDs) [16].
1Project page: https://recommendation-as-generation.github.io/
∗Equal contribution.
§Work done during an internship at Kuaishou Technology.
†Corresponding author.
1.UserProfileGender:femaleAge:18Preference:#pretty selfies#love songs#learntomakeup2.UserBehavior
#milk tea & selfies#lovemusic
#makeupbrands#beauty& travel
predictinterestSIDsgeneratepersonalizedvideosGRM
=【Pretty girl】【makeup tutorial】with 【emotionalmusic】and 【a classy vibe】interestalignmentfeedbackoptimization
!
!
"
predictinterestvideoidsretrievetopvideosDLRMsinsensitivetouserinterestinsensitivetouserfeedback
"
【popstar】
【music】
【milktea】
【dog】
【holiday】
(a)
(b)RecommendationParadigmShift:fromRetrievaltoVideoGenerationpre-producedvideos
interest-driven personalized video generation
Figure 1:Recommendation paradigm shift. (a) DLRMs retrieve
videos from a fixed content pool, leading to suboptimal matches
when user interests fall outside the pool; (b) Our paradigm gener-
ates personalized videos on demand that both align with the user
interests predicted by a GRM and are driven by real user feedback
in a closed loop, breaking the fixed-pool limit.
Despite these advances, existing systems remain fundamentally
constrained by a static pool of pre-produced videos. Recommenda-
tion models can only retrieve the best available content from the
existing pool, even when user interests fall outside the pool. This
limitation is particularly severe in modern short-video platforms,
where user interests are more dynamic, long-tailed, and diverse. As
a result, retrieval-based systems are inherently limited in faithfully
capturing fine-grained user interest.
Meanwhile, recent breakthroughs in AI-generated content (AIGC)
[8,15,17,20,22] have demonstrated unprecedented capabilities
in open-domain video creation. Modern video generation models
can produce cinematic-quality visual content with strong semantic
controllability, opening up a new opportunity for recommendation:
Can recommendation systems move beyond retrieving existing
videos to directly generate personalized videos from inferred user
interests?
Answering this question requires addressing two key challenges
in recommendation and generation systems.
The first challenge is how to bridge recommendation and
generation into a unified framework.Recommendation mod-
els are trained on heterogeneous and discrete data, including user
profiles, item features, and user behaviors, aiming to predict user
interests. In contrast, video generation models operate on multi-
modal continuous signals, such as text, images, audio, and motion,arXiv:2606.25496v1  [cs.IR]  24 Jun 2026

Yanhua Cheng et al.
focusing on generating coherent and high-fidelity videos. Given
such fundamental differences in data representation and learning
objectives, recommendation and generation are typically developed
as two separate tasks, making it difficult to translate predicted user
interests into controllable video generation. This separation also
blocks user feedback from flowing back into the generation pro-
cess, limiting the diversity and interest-alignment of the produced
content.
The second challenge is how to generate high-quality and
interest-aligned videos at industrial scale.Although recent
state-of-the-art video generation models [ 8,15,17,20,22] achieve
strong visual quality, they remain difficult to deploy in large-scale
recommendation systems. These models often rely on manual
prompting, multi-stage refinement and post-processing with pro-
fessional tools, resulting in high latency and computational cost to
produce a single user-satisfactory video. Personalizing across the
diverse and long-tailed interests of hundreds of millions of users
further amplifies these costs, making it infeasible to deploy such
models directly in production.
To address these challenges, we proposeRecommendation-as-
Generation(RaG), a new paradigm that unifies recommendation
and personalized video generation in a closed-loop framework, as
illustrated in Figure 1. Instead of retrieving from a fixed pool, RaG
generates personalized videos directly from inferred user interests.
A key idea of RaG is to useDisentangled Semantic IDs(D-
SIDs) as a unified interface between recommendation and gen-
eration. A multimodal large language model encodes each video
into two factorized embeddings—one forcontent(entities, topics)
and the other forcreativeattributes (style, rhythm, atmosphere).
These embeddings are then quantized into discretecontent SIDs
andcreative SIDs, jointly forming the video’s D-SIDs. On the rec-
ommendation side, a generative recommendation model (GRM)
autoregressively predicts the D-SIDs of user interests. On the gen-
eration side, the predicted D-SIDs are decoded into personalized
videos, connecting fine-grained interest modeling with controllable
video generation.
To realize controllable video generation at scale, RaG develops
Video Generation Agents(VGAs). Compared to monolithic, high-
cost diffusion-based or prompt-engineering-heavy pipelines, VGAs
adopt a hierarchical planning and refinement framework. Condi-
tioned on user-interest D-SIDs, a fine-tunedInstruction Model
(IM) first translates them into structured generation blueprints.
Three role-specialized agents then reason and act over the blueprints,
jointly modeling visual composition, audio alignment, and artis-
tic effects. The three agents share a single LLM backbone and are
jointly trained end-to-end, differentiated only through prompts
and tool access. After the agents complete the pipeline, a bounded
reflection loop (capped at two iterations) refines cross-modal con-
sistency, balancing output quality with generation efficiency. The
shared backbone further enables KV-cache reuse across agents to
substantially accelerate inference. Combined with an SID-indexed
cache that amortizes generation cost, VGAs reliably serve recom-
mendation requests for hundreds of millions of users at industrial
scale.
To close the optimization loop, RaG introducesSynergistic
Cross-Domain Reward Learning(SCRL). Instead of naive re-
ward aggregation that conflates heterogeneous reward signals,SCRL formulates multi-objective optimization as a constrained
policy learning problem: user feedback serves as the primary objec-
tive, while interest alignment and video quality act as constraints.
Group-decoupled reward normalization (GDPO [ 10]) is applied per
channel to reconcile scale mismatch, followed by a PID-controlled
Lagrangian update [ 19] to stabilize training. Together, SCRL unifies
recommendation and video generation into a single closed-loop
optimization where user interests, content quality, and real-world
feedback co-evolve.
We deploy RaG on a large-scale production platform serving
over 400 million daily active users in a revenue-critical advertising
scenario. Online A/B testing shows significant improvements in ad
revenue, validating the effectiveness of generation-driven person-
alization for recommendation. To the best of our knowledge, this
is the first production-scale system that effectively unifies recom-
mendation and personalized video generation.
Our main contributions are summarized as follows:
•We proposeRecommendation-as-Generation(RaG), a new
paradigm that shifts recommendation from retrieving videos
within a fixed pool to generating personalized videos directly
from inferred user interests. Disentangled Semantic IDs (D-SIDs)
serve as the unified latent interface between recommendation
and generation, and Synergistic Cross-Domain Reward Learning
(SCRL) closes the loop by enforcing interest alignment, user
feedback, and video quality assessment.
•We develop industrial-scaleVideo Generation Agents(VGAs)
with hierarchical planning, collaborative multi-agent execution,
and iterative refinement, enabling scalable and high-quality per-
sonalized video production.
•Extensive offline experiments and online A/B testing on a pro-
duction platform demonstrate substantial improvements in ad
revenue, validating the effectiveness of large-scale personalized
video generation for recommendation.
2 Methodology
2.1 Paradigm Shift: Recommendation as
Generation
Conventional recommendation systems [ 3,5,34,36] retrieve or
rank videos from a fixed content pool. Recent generative recom-
mendation models (GRMs) formulate recommendation as autore-
gressive token prediction [ 7,27], but still retrieve videos from the
existing content pool set according to the predicted tokens. As a
result, these approaches remain limited by content coverage, often
yielding suboptimal recommendations when user interests involve
novel or long-tail semantics.
To overcome this limitation, we introduce theRecommendation-
as-Generation(RaG) paradigm, which reformulates recommenda-
tion as an interest-conditioned video generation problem (Figure 2).
Instead of retrieving existing videos, RaG directly generates person-
alized videos from inferred user interests. One key idea is to unify
recommendation and video generation within a shared discrete
latent space.
We construct this space usingDisentangled Semantic Video
Encoders(Section 2.2), which map videos into disentangled seman-
tic IDs (D-SIDs). These D-SIDs capture both semantic content and

Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale
RewardsfromUserFeedbackGenerativeRecommendationModel
userstaticprofile
sparsereal-worldfeedbacksemanticIDscontentinterestcreativeinterestInstructionModel
semanticIDsfromGRM(optionalmetadata)detailedinstructionsforvideogenerationcontentinstructioncreativeinstruction</think>…Given that the only available asset is a model wearing the outfit, generate a close-up product shot that emphasizes premium texture and detailsofthe garment, presenting it as a high-end gift. …</think>VisualPlanningAgent
cumulativecontext&visualtoolmaskvisualrolepromptAudioAlignmentAgent
cumulativecontext&audiotoolmaskaudiorolepromptArtisticEffectEnhancementAgent
cumulativecontext&effecttoolmaskeffectroleprompt
</action>…Hereweneedtocallthetext2videotooltogeneratethevideoclipthat…</action></think>…The product is women’s clothing, with a female model and a primarily female target audience. Use a female voice for TTS, and select background music that is light, cheerful, and pleasant. …</think></think>…Finally, add promotional stickers, tags, and discount visuals to highlight the product’s value for money, along with a call-to-action button to encourage users to place an order. …</think></action>…Hereweneedtocallthettsandbgmtooltoembed audio cuesthat…</action>
</action>…Hereweneedtocalltheartistic typography tooland the button tooltohighlightthat…</action>VideoGenerationAgentsDisentangledSemanticVideoEncoders
inputvideo(title/OCR/ASR/visualframes)
LOOPoutputgeneratedvideos
 visualframeembeddings/MLLMwithLoRA
contentorcreativecaptions/embeddingscontentembedding/semanticIDscreativeembedding/semanticIDsRQ-KmeansEncoder
/contentorcreativeembeddingssemanticIDs/
Synergistic Cross-Domain Reward LearningRewards of Interest AlignmentRewardsofVideoQualityAssessment
generatedvideo
!
⭐
#likecollectpurchaseRankingModelhistoricalbehaviors
!
⭐
#likecollectpurchasedensepredictedfeedbacksemanticIDsofGRMgeneratedinstructionsofIMsemanticsimilaritysemanticIDsofGRMgeneratedvideoframessemanticsimilarityinterestalignmentofinstructionsinterestalignmentofgeneratedvideovisualqualityscore
generatedvideoframes
generatedtts&bgmaudio-visualcoherencegeneratedeffects
$
%
&subtitleshighlightwordsactionbarseffect-visualalignment
Figure 2:Architecture of the Recommendation-as-Generation (RaG) framework. Videos are encoded into Disentangled Semantic IDs (D-
SIDs) that decouple content and creative semantics, forming a shared latent interface for recommendation and generation. The Generative
Recommendation Model (GRM) predicts a user’s interest D-SIDs from user context. The Instruction Model (IM) then translates these predicted
D-SIDs, together with optional metadata, into shot-level production instructions, which are executed by the Video Generation Agents (VGAs)
through hierarchical planning and refinement. The full pipeline is jointly optimized under Synergistic Cross-Domain Reward Learning (SCRL).
creative attributes, enabling both fine-grained interest modeling
and controllable video generation. Given a video 𝑣, the encoderE
produces a sequence of tokens:
D−SIDs=E(𝑣)=(𝑠1
content,...,𝑠𝐿
content,𝑠1
creative,...,𝑠𝐿
creative),(1)
which jointly represent video semantics and creative structure.
Building on this semantic space, recommendation is recast as
generative interest modeling: given a user’s profile and interaction
history, the Generative Recommendation Model (GRM, Appendix C)
autoregressively predicts the sequential D-SIDs representing the
user’s future interests:
𝑝(D−SIDs|c user)=2𝐿Ö
𝑡=1𝑝(𝑠𝑡|𝑠<𝑡,cuser),(2)
wherec userdenotes the user context.
Unlike prior GRM-based approaches that use predicted D-SIDs as
retrieval keys, we treat D-SIDs asgenerative interest representations
that can be directly decoded into new content, beyond a fixed pool.
The overall pipeline is:
D−SIDs=E(𝑣)→𝑝(D−SIDs|c user)→ ˆ𝑣=G(D−SIDs),(3)
where user interests are modeled in the latent semantic space and
decoded into personalized videos. However, directly optimizing G
for both generation quality and interest alignment is challenging.
We therefore decompose the generation process into a hierarchical
framework.
We introduce anInstruction Model(Section 2.3) that translates
D-SIDs into natural language instructions, providing interpretable
and structured guidance for downstream agents. Building on this,
we developVideo Generation Agents(Section 2.4) that generatevideos through collaborative agents, enabling hierarchical planning,
multimodal alignment, artistic enhancement, and iterative refine-
ment. Finally, we optimize the entire framework viaSynergistic
Cross-Domain Reward Learning(Section 2.5), jointly capturing
user interest alignment, generation quality, and user engagement
signals.
2.2 Disentangled Semantic Video Encoders
2.2.1 Multimodal Representation Learning.Building upon Qwen2.5-
VL-7B-Instruct [ 21], we propose an instruction-guided disentan-
gled representation framework that separates semantic content
and creative attributes from the same video. For multimodal input
processing, we directly reuse Qwen2.5-VL’s native visual encoder
and text tokenizer.
We first extract its visual token representations using the vision
encoder:𝐻=F(𝑣),𝐻∈R𝑁×𝑑,where𝐻denotes a sequence of
visual tokens capturing spatial-temporal semantics.
To obtain disentangled signals, we leverage our in-house dense
captioning model (CapModel) to generate factor-specific textual
descriptions:
𝐷𝑚=CapModel(𝑣,PROMPT 𝑚), 𝑚∈{content,creative},(4)
where𝐷content describes semantic content (entities, topics), while
𝐷creative captures creative attributes (style, rhythm and atmosphere).
The instructions are encoded via the text tokenizer: 𝑄𝑚=T(𝐷𝑚),
𝑄𝑚∈R𝐿𝑚×𝑑, where𝐿𝑚is the instruction length. We obtain multi-
modal representations by jointly encoding visual and textual inputs
with Qwen2.5-VL-7B-Instruct, and use the last-token hidden state

Yanhua Cheng et al.
of the final layer as the pooled multimodal representation:
z𝑚=Normalize(VLM(𝐻,𝑄 𝑚)),z𝑚∈R𝑑,∥z𝑚∥2=1,(5)
yieldingz content andz creative as L2-normalized content and creative
representations, respectively.
To encourage representation consistency, we employ a con-
trastive loss for each module:
L𝑚=−logexp(sim(z𝑖
𝑚,z𝑗
𝑚)/𝜏)
Í
𝑘exp(sim(z𝑖𝑚,z𝑘𝑚)/𝜏),(6)
wherez𝑗
𝑚is the positive pair ofz𝑖
𝑚within a batch, and k indexes
all candidates including the positive.
To reduce cross-factor leakage, we introduce an orthogonality
constraint:
Lorth=∥z⊤
content zcreative∥2
2.(7)
The final objective is:
L=L content+𝛾1Lcreative+𝛾2Lorth.(8)
2.2.2 Discrete Tokenization.To facilitate generative recommenda-
tion within the latent space, we discretize the disentangled multi-
modal representations into semantic IDs.
Specifically, each representationz 𝑚is independently quantized
via Residual Quantization (RQ)-based K-means [ 11], yielding a quan-
tized embeddinge 𝑚that approximatesz 𝑚as a sum of codebook
vectors across𝐿hierarchical layers:
e𝑚=𝐿∑︁
𝑙=1c𝑙
𝑚(𝑠𝑙
𝑚)≈z𝑚,e𝑚∈R𝑑,(9)
where𝑠𝑙
𝑚denotes the discrete code index at layer 𝑙for modality 𝑚,
andc𝑙
𝑚(·)is the corresponding codebook lookup. Each modality
maintains an independent codebook with 8,192 entries per layer.
The final D-SIDs are obtained by concatenating the per-modality
code sequences:D-SIDs=
𝑠1:𝐿
content ;𝑠1:𝐿
creative
.
2.3 Instruction Model
The Instruction Model translates disentangled semantic IDs into
shot-level video production instructions. Unlike conventional cap-
tion generation, these instructions explicitly specify scene composi-
tion, camera motion, temporal pacing, and cinematic style, serving
as an intermediate semantic bridge between discrete user interests
and controllable video generation.
2.3.1 Supervision Construction.Since no off-the-shelf dataset con-
tains video-instruction pairs at the shot level, we distill supervision
from a strong multimodal teacher. For each video 𝑣, we first extract
its D-SIDs (Section 2.2), and then prompt Gemini2.5 Pro [ 4] with
a carefully designed instruction template PROMPT instto produce
the target shot-level script:
𝐷inst=Gemini(𝑣,PROMPT inst)=(𝑦 1,𝑦2, ...,𝑦𝐿inst),(10)
where𝐷instis a token sequence of length 𝐿instserving as ground-
truth supervision. To accommodate advertising scenarios where the
generated video must reflect specific products being promoted, we
further introduce anoptionalmetadata factor 𝐷meta(e.g., product
information and marketing topics) as an auxiliary conditioning
signal. When unavailable (e.g., for pure organic videos), 𝐷metaissimply masked, leaving instruction generation conditioned on D-
SIDs alone.
2.3.2 Model and Optimization Objective.We instantiate the In-
struction Model with Qwen3-8B [ 29], which consumes two hetero-
geneous token sequences—the primary D-SIDs and the auxiliary
𝐷meta(masked when unavailable)—mapped into the LLM’s input
embedding space and concatenated as the prefix.
For D-SIDs, we reconstruct continuous embeddings from the dis-
crete codes via reverse RQ-Kmeans,e D-SIDs =[econtent ;ecreative]∈
R2×𝑑, and map them through a learnable projector 𝜙(·) toh D-SIDs =
𝜙(eD-SIDs)∈R2×𝑑′. For metadata, 𝐷metais tokenized and embed-
ded by the LLM’s native text tokenizer Tto𝑄meta=T(𝐷 meta)∈
R𝐿meta×𝑑′, with𝐿metadenoting the token length. Conditioned on
both, the model autoregressively predicts the instruction sequence
ˆ𝐷inst=LLM(h D-SIDs,𝑄meta),(11)
and is optimized with a standard next-token prediction loss against
the Gemini-distilled supervision:
LNTP=−𝐿inst∑︁
𝑡=1log𝑃 𝑦𝑡𝑦<𝑡,hD-SIDs,𝑄meta,(12)
so that ˆ𝐷insttoken-wise approximates𝐷 inst.
2.3.3 Three-Stage Training.We adopt a three-stage training strat-
egy: in the first stage, the backbone LLM is frozen and only the
projector𝜙(·) is optimized to align D-SIDs’ embeddings with the
language space; in the second stage, both the projector and LLM
parameters are jointly fine-tuned for improved semantic fidelity
and controllable instruction generation; in the third stage, we fur-
ther enhance the model using reinforcement learning with reward
optimization, as described in Section 2.5.
2.4 Video Generation Agents
As illustrated in Figure 2, industrial-scale personalized video gen-
eration cannot be effectively handled by a monolithic generator.
The one-shot production of visuals, audio, and effects often leads to
semantic inconsistency and limited controllability. Moreover, video
production exhibits a hierarchical dependency structure where vi-
sual planning determines the narrative flow, while audio and effects
are conditioned on the visual state.
To address this, we propose Video Generation Agents (VGAs),
formulated as a structured multi-agent decision process over an
evolving generation state.
Agentic Formulation.We model video generation as a sequential
decision process executed by a team of sub-agents. At each step
𝑡, the active sub-agent observes a state S𝑡, selects an action 𝑎𝑡
according to its policy 𝜋𝜃, and transitions to the next state via a
deterministic operatorP:
𝑎𝑡∼𝜋𝜃(𝑎𝑡|S𝑡),S𝑡+1=P(S𝑡,𝑎𝑡).(13)
To enable efficient backbone reuse across sub-agents (detailed later),
we serialize the state as an ordered prefix followed by stage-dependent
tokens:
S𝑡=ˆ𝐷inst;𝐷tool|       {z       }
shared prefix;O<𝑡;PROMPT role|                 {z                 }
stage-dependent
,(14)

Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale
where ˆ𝐷instis the instruction sequence produced by the Instruc-
tion Model; 𝐷toolis the description of all available tools, including
in-house pretrained text-to-video and image-to-video models and
external audio synthesis and visual effect APIs; O<𝑡is the running
concatenation of all earlier sub-agents’ role prompts and their gen-
erated outputs; and PROMPT roleis a short role-specific prompt that
activates the current sub-agent. The action 𝑎𝑡corresponds to a
modality-specific intent for visual, audio, or effect generation.
VGAs consist of three role-specialized sub-agents, each acting
according to its own policy:
𝜋𝜃(𝑎𝑡|S𝑡)={𝜋 visual, 𝜋audio, 𝜋effect}(𝑎𝑡|S𝑡),(15)
corresponding to visual planning, audio synthesis, and post-production
effects, respectively. We next describe each sub-agent in turn.
1. Visual Planning Agent (VPA).At the visual stage, PROMPT role=
PROMPT visual andO<𝑡is empty (or carries the previous reflection
round’s content). The VPA acts as the global controller, producing
a clip-level storyboard with scene segments, layout configurations,
and temporal boundaries:I visual=𝜋 visual(S𝑡).
2. Audio Alignment Agent (AAA).At the audio stage, PROMPT role
=PROMPT audio andO<𝑡extends with(PROMPT visual,Ivisual). The
AAA generates temporally aligned audio (speech and music) syn-
chronized with scene transitions:I audio=𝜋 audio(S𝑡).
3. Artistic Effect Enhancement Agent (AEEA).At the effect
stage, PROMPT role=PROMPT effect andO<𝑡further extends with
(PROMPT audio,Iaudio). The AEEA performs post-production refine-
ment by adding subtitles, visual effects, transitions, and call-to-
action elements:I effect=𝜋 effect(S𝑡).
Hierarchical Generation with Bounded Reflection.The three
intent outputs are composed into the final video via a unified gen-
eration operatorG:
V=G(I visual,Iaudio,Ieffect).(16)
To improve cross-modal consistency, VGAs operate within a bounded
reflection loop that follows the standard Observe →Think→Act
cycle, capped at two iterations to balance output quality with gen-
eration efficiency.
Shared Backbone and KV-Cache Reuse.Although VGAs com-
prise three role-specific sub-agents, they are not three separate
models—all share a single Qwen2.5-32B backbone [ 28] with fully
shared parameters. Differentiation arises purely from the state S𝑡
in Eq. (14):PROMPT roleactivates the target sub-agent, an atten-
tion mask over 𝐷toolrestricts it to the accessible tool subset, and
O<𝑡supplies the cumulative upstream context. This serialization
also enables straightforward KV-cache reuse: with sub-agents in-
voked sequentially and O<𝑡growing append-only, every previously
generated token stays in the KV cache, leaving each downstream
sub-agent to only encode its own PROMPT role, substantially reduc-
ing per-request inference latency.
The agent policies are optimized via synergistic cross-domain
reward signals that jointly capture generation quality, interest align-
ment, and user feedback, as detailed in Section 2.5.
2.5 Synergistic Cross-Domain Reward Learning
2.5.1 Cross-Domain Reward Formulation.To further enhance both
recommendation and video generation performance, we formulatea structured cross-domain reward scheme with three synergistic ob-
jectives:video quality,interest alignment, anduser feedback. Unless
otherwise specified, all reward models share the same Transformer-
based architecture and are trained on task-specific datasets.
1. Video Quality Rewards.
We evaluate the perceptual and compositional quality of gen-
erated videos from three complementary aspects: visual quality,
audio-visual coherence, and effect-visual alignment. Formally, we
define:𝑅 quality =𝑅 visual+𝑅 audio+𝑅 effect, where:
•𝑅 visual evaluates visual quality, including aesthetic appeal and
spatio-temporal consistency to ensure coherent motion and sta-
ble rendering,
•𝑅 audio measures alignment between audio and visual content,
covering both speech synchronization (TTS) and background
music consistency (BGM),
•𝑅 effect captures the quality and alignment of visual effects, includ-
ing subtitles, highlights, and interactive elements such as action
bars.
2. Interest Alignment Rewards.
To keep generated content aligned with user interests through-
out the pipeline, we apply alignment rewards at multiple stages,
anchored on the D-SIDs that encode user interests in a structured
latent space:𝑅 align=𝑅 instr-align+𝑅 rep-align , where:
•𝑅 instr-align enforces semantic consistency between GRM-generated
D-SIDs and the generated instructions,
•𝑅 rep-align measures semantic similarity between GRM-generated
D-SIDs and the generated videos.
3. User Feedback Rewards.
To enhance downstream user engagement, we leverage user
interaction signals such as clicks and conversions as the core reward
for optimization. However, real-world interaction signals are sparse
and delayed, making them insufficient for stable and efficient policy
optimization.
To mitigate this issue, we augment sparse interaction signals
with dense engagement estimates from deployed ranking models.
We define the overall reward as:𝑅 feedback =𝑅 real+𝑅 pred, where:
•𝑅 realdenotes sparse but high-fidelity user interaction signals
observed from real feedback, including behaviors such as click,
like, collect, and purchase,
•𝑅 preddenotes dense engagement signals estimated by ranking
models, which capture user preference strength beyond explicit
interactions.
2.5.2 Constrained Policy Optimization with GDPO.To jointly opti-
mize the heterogeneous, cross-domain rewards introduced above,
we formulate reward learning within the RaG framework as acon-
strained policy optimizationproblem solved by GDPO [ 10]. The de-
sign addresses two practical challenges in multi-reward RL: (i) the
scale mismatch and optimization instability caused by heteroge-
neous reward signals, and (ii) the difficulty of statically balancing
competing objectives without sacrificing the dominant goal.
Problem setup.Given an input context 𝑥, the policy 𝜋𝜃samples
a candidate setY={𝑦 1,...,𝑦𝐾}∼𝜋𝜃(·|𝑥) . Each candidate 𝑦𝑖
is evaluated by a collection of heterogeneous reward functions
covering user feedback 𝑅feedback(𝑦𝑖), interest alignment 𝑅align(𝑦𝑖),

Yanhua Cheng et al.
GenerativeRecommendationModelInstructionModelVideoGenerationAgents
CachedResultscache hit via contentSIDs & creativeSIDscache expansionwith generatedpersonalizedvideosrequestconsumeServingSystemLogCollect&PreprocessOfflineModelTrainingGRM/IM/VGAsGDPOSamplerServerGRM/IM/VGAsCross-DomainRewardModelsTrainingSystemmodelparametersyncGDPOServermodelparametersyncrolloutresponserewardrollout trajectories
Figure 3:Training and serving architecture of the proposed
Recommendation-as-Generation system.
and video quality 𝑅quality(𝑦𝑖). These rewards differ in scale, density,
and reliability, motivating the constrained formulation below.
Constrained reward formulation.We designate user feedback
as the primary objective and treat interest alignment and video
quality as inequality constraints with target thresholds 𝜏𝑎(𝑙𝑖𝑔𝑛) and
𝜏𝑞(𝑢𝑎𝑙𝑖𝑡𝑦) . The composite reward for each candidate 𝑦𝑖is defined as
𝑅(𝑦𝑖)=𝑅 feedback(𝑦𝑖)−∑︁
𝑐∈{𝑎,𝑞}𝜆𝑐(𝑡)ReLU 𝜏𝑐−𝑅𝑐(𝑦𝑖),(17)
where𝜆𝑎(𝑡),𝜆𝑞(𝑡)≥ 0are time-varying Lagrangian multipliers,
updated via a PID-controlled rule on constraint violations [ 19] to
avoid the oscillation and overshoot of naive primal–dual updates.
To avoid hand-tuned magic numbers, we calibrate each threshold
relative to the SFT baseline distribution on a held-out validation
set as𝜏𝑐=𝜇base
𝑐+𝑘𝑐𝜎base
𝑐, where the strictness factor 𝑘𝑐encodes
the module’s role in RaG: VGAs adopt the strictest setting ( 𝑘𝑐=1.1
for both𝜏𝑎and𝜏𝑞) as it directly governs final video generation;
IM retains a comparable 𝜏𝑎(𝑘𝑎=0.8) to enforce instruction-level
alignment; while GRM applies a relaxed 𝜏𝑎(𝑘𝑎=0.3), with the
video-quality constraint omitted for the latter two modules.
Group-decoupled normalization and advantage.Given the con-
strained reward in Eq. (17), GDPO further eliminates residual scale
mismatch among reward channels via per-reward standardization
prior to aggregation, and computes a group-relative advantage over
the sampled candidate setY:
𝐴𝑖=𝑅(𝑦𝑖)−𝜇(Y)
𝜎(Y)+𝜖,(18)
where𝜇(Y) and𝜎(Y) denote the group-level mean and standard
deviation of the rewards over Y. This decoupled normalization
stabilizes optimization across rewards with disparate magnitudes.
Optimization objective.The policy is updated by maximizing the
group-relative advantage anchored to the frozen SFT policy𝜋 ref:
LGDPO=−E(𝑥,𝑦 𝑖)
𝐴𝑖log𝜋𝜃(𝑦𝑖|𝑥)
𝜋ref(𝑦𝑖|𝑥)
.(19)
For brevity, we omit the importance-sampling ratio clipping and the
KL regularization term against 𝜋refthat are commonly used to sta-
bilize policy optimization; both are retained in our implementation
and follow the standard GDPO formulation [10].
3 Deployment
We deploy RaG in Kuaishou’s large-scale advertising system, serv-
ing over 400 million users (Figure 3). The system unifies real-timeuser interest modeling with large-scale personalized video gener-
ation under strict latency constraints. Since video generation is
orders of magnitude slower than interest inference (Appendix A),
we design a decoupled deployment architecture to bridge this effi-
ciency gap while maintaining end-to-end personalization quality.
The system consists of three decoupled modules: real-time interest
modeling, nearline video generation, and latency-aware serving.
Real-Time Interest Modeling.The Generative Recommendation
Model (GRM) is continuously trained on streaming user interaction
logs (impression, click, watch time, and conversion) to adapt to non-
stationary user behavior, combining streaming supervised updates
with periodic GDPO-based optimization.
At real-time inference, GRM performs low-latency autoregres-
sive generation of structured Semantic IDs (SIDs), which encode
user interests as semantic targets for downstream content genera-
tion.
Nearline Video Generation.The Instruction Model (IM) and
Video Generation Agents (VGAs) are trained on large-scale agentic
supervision data curated from high-quality videos, and optimized
via supervised fine-tuning followed by constrained GDPO to jointly
improve generation quality and interest alignment. Both models
are periodically updated in full-batch mode to ensure training sta-
bility while adapting to evolving user interests and emerging video
patterns.
At serving time, conditioned on GRM-generated SIDs, IM and
VGAs operate in a nearline pipeline to generate personalized videos.
To handle the heavy generation load, VGAs applyKV-cache reuse
as established in Section 2.4: with sub-agents invoked sequen-
tially over an append-only state, every previously generated to-
ken stays in the KV cache, leaving only each sub-agent’s own
short PROMPT roleto be encoded per call, substantially reducing
per-request inference latency. The outputs are continuously accu-
mulated into a growing personalized video space, enabling coverage
expansion while decoupling video generation from real-time serv-
ing.
Latency-Aware Serving.To meet real-time consumption require-
ments in recommendation scenarios, the system adopts a hierar-
chical serving strategy organized around whether the requested
content-level SIDs are covered by the cache.
Case 1: content-SIDs hit.If the matched cache entry also covers
the creative-level SIDs, the system returns the previously gener-
ated video directly with negligible latency; otherwise, it serves a
content-consistent cached video while asynchronously scheduling
the missing creative variations, with higher-frequency creatives
prioritized in the generation queue.
Case 2: content-SIDs miss.The system serves videos associated
with the nearest-neighbor SIDs for immediate consumption, while
enqueuing the uncovered SIDs for prioritized future generation.
4 Experiments
4.1 Online A/B Testing
We deploy the proposed Recommendation-as-Generation (RaG)
framework in the real-world advertising platform of Kuaishou and
conduct large-scale online A/B experiments to evaluate its indus-
trial effectiveness. The experiments mainly focus on two aspects:

Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale
Table 1:Online A/B test results. Rev. denotes ad revenue. Results
are reported as relative improvements over production baselines.
MethodRev. (%↑)
vs. DLRM baselineRev. (%↑)
vs. GRM baseline
Production Baseline
DLRM baseline – –
GRM baseline [27] +3.526% –
Enhanced GRM
GRM + Disentangled-SIDs (D-SIDs) +4.460% +0.902%
Full System (RaG)
RaG (GRM + D-SIDs + IM + VGAs + SCRL)+5.462% +1.870%
Table 2:Quality of the Disentangled SIDs. We report both (i)
embedding-based semantic retrieval quality and (ii) SID discretiza-
tion quality. Improvements over the strongest baseline are high-
lighted.Impr.: improvement;R@k: Recall@k;Cpr.: compression
rate;Col.: collision rate.
Method Semantic Retrieval (R@K) Discretization Quality
R@1↑R@5↑R@10↑Cpr.↓/Col.↓
VLM2Vec-V2 [12] 0.485 0.690 0.756 –
QARM [11] 0.541 0.812 0.893 1.14 / 18.24%
Qwen2.5-VL-7B [32] 0.769 0.948 0.977 –
Ours (D-SIDs) 0.896 0.985 0.994 1.02 / 2.62%
Impr.+16.5% +3.9% +1.7% -10.5% / -15.6pp
(1)the effectiveness of Disentangled Semantic IDs (D-SIDs) for
generative recommendation, and(2)the additional gains brought
by SID-driven personalized video generation.
Table 1 summarizes the online results. Replacing the production
DLRM-based pipeline with the Generative Recommendation Model
(GRM) yields consistent ad revenue gains, and the proposed Disen-
tangled Semantic IDs (D-SIDs) further lift the improvement from
+3.526% to +4.460%, confirming that decoupling content and cre-
ative semantics yields a more structured latent space and mitigates
interference during autoregressive generation. Nevertheless, both
variants remain within the retrieval paradigm, selecting candidates
from a fixed pool.
Finally, the full RaG framework—integrating GRM, D-SIDs, the
Instruction Model (IM), and Video Generation Agents (VGAs) un-
der Synergistic Cross-Domain Reward Learning (SCRL)—delivers a
+5.462% ad revenue gain over the DLRM-based pipeline. Crucially,
RaG also outperforms the strong GRM baseline by +1.870%, with
this additional lift coming directly from D-SIDs-driven personalized
video generation. This marks a paradigm shift from retrieval-based
to generation-based recommendation, where user interests actively
drive personalized content production rather than merely matching
existing candidates.
4.2 Offline Ablation Studies
We ablate the key components of RaG framework—D-SIDs, IM,
VGAs, and SCRL optimization—to assess their contributions in
terms of semantic representation quality, instruction generation
capability, and reward-driven video generation performance.
4.2.1 Quality of Disentangled SIDs.The D-SIDs consist of two core
components, i.e., multimodal representation learning and semantic
quantization; we systematically analyze their effectiveness in the
following experiments.Table 3:Evaluation of videos between the proposed VGAs vs. the
workflow baseline. For Automated Score, we present average and
median score.
Metric Workflow Baseline VGAsImpr.
Automated Score↑62.4 / 62.071.3 / 76.0 +14.3% / +22.6%
Automated Win Rate↑28.7%70.1% +41.4pp
User Study Win Rate↑34.4%52.9% +18.5pp
Table 4:Reward ablation with corresponding evaluation metrics.
For each reward component, we report its dedicated metric on a
corresponding evaluation set, comparing the policy trained with
that reward (Ours) against the no-reward base policy (Base).
Video Quality Rewards Automated Win Rate↑
𝑅visual𝑅audio𝑅effect Base→Ours
✓29.3%→50.7% (+21.4pp)
✓24.0%→48.0% (+24.0pp)
✓22.7%→41.3% (+18.6pp)
✓ ✓ ✓37.3%→56.0% (+18.7pp)
+ Interest Alignment Rewards Interest Alignment Score↑
𝑅align Base→Ours
✓0.707→0.828 (+17.1%)
Multimodal Representation Learning.We evaluate the pro-
posed instruction-guided representation learning under a product-
level retrieval setting to ensure fair comparison of semantic align-
ment capability. As shown in Table 2, our method consistently out-
performs all baselines, achieving0 .896/0.985/0.994in R@1/5/10.
In particular, R@1 improves by +16.5%over the strongest baseline
(Qwen2.5-VL-7B), demonstrating stronger semantic discriminabil-
ity under identical retrieval conditions.
Semantic Quantization.We construct the D-SIDs by applying RQ-
KMeans residual quantization separately to the content and creative
embeddings, yielding disentangled content SIDs and creative SIDs.
For a fair comparison, both D-SIDs and QARM adopt an identical
quantization setup with a 4-layer codebook and 8,192 codes per
layer. As shown in Table 2, our method achieves superior discretiza-
tion quality, reducing compression distortion to1.02and collision
rate to2.62%. Compared to QARM (1 .14/18.24%), this corresponds to
a10.5%reduction in compression error and a15 .6pp lower collision
rate, indicating a more compact and collision-resistant semantic
space.
4.2.2 Instruction Model Configuration.We evaluate the Instruc-
tion Model in terms of decoding fidelity, measured by the cosine
similarity between generated instructions and ground-truth sum-
maries using Qwen3-Embedding-8B. Empirically, we observe that
both increased training data and model capacity lead to consistent
improvements. Specifically, the performance improves from0 .7760
(8B model, 100K samples) to0 .8096(8B model, 1M samples), and
further to0.8212with a 32B model trained on 1M samples.
Considering the trade-off between performance and computa-
tional efficiency, we adopt the 8B model trained on 1M samples as
the default configuration, which achieves competitive decoding fi-
delity while offering significantly lower deployment cost compared
to larger models.

Yanhua Cheng et al.
GRM VGAs•Gender : female
•Age: 25-34
•Preference : young mother lifestyl
e, baby care, household essentials, 
value -for-money shopping
IMUser Profile
User History - Marketing Topic
China's leading 
authenticated lifestyle e -
commerce platform. 
Discover curated quality 
goods at competitive 
prices — one platform, one -
stop shopping, zero doubt.
 - Product InformationMetadataEmbedding
Content
Creative-Shot -Level Description
It opens with a scene of ayoung 
mother in a cozy living room … 
The sequence then shifts to a 
close -up shot at a modern 
minimalist vanity … 
Subsequently, a softly colored 
children‘s room scene is 
interspersed  … The video 
concludes with a freeze -frame of 
a young family  of three standing 
behind Poizon  gift boxes ...
-Creative Style
Throughout, the video employs 
a narrative arc of unboxing 
surprise → efficacy 
demonstration → parent -child 
warmth → influencer 
endorsement → harvest 
visualization → brand freeze -
frame …Detailed Instructions
Visual Track
-shot 1:{
“caption”:“Theyoung mother
excitedly unboxes theblue
delivery package, holding
upabeautifully packaged
setofbeauty andskincare
products, eyes filled with
anticipation …”,
“duration”:5.2
}
…
-shot N:{…}Audio Track
-dialogue
{
“text”:“Skincare products
must have anauthenticity …”,
“tts_type”:“Female -Young -
Gentle”,
}
-BGM :{
“type”:“Cheerful -Pop”,
“resource”:“Medium -xxx.mp3”,
}Effect Track
-subtitle
{
“text”:“Skincare products
must have anauthenticity …”,
“pos”:[50,75],“time”:[0.0, 5.2],
“font”:“xx.ttf“,“style”:“xx.zip”
}
-xfx:[{
“type”:“pop_surprise  sound”,
    “time”: [5.2, 6.2]}, …
   ]
Generated VideoGenerated Track InstructionsPreprocessing
D-SIDs
Figure 4:Qualitative example of interest-driven personalized video generation in advertising scenarios.
4.2.3 Performance Analysis of Video Generation.We evaluate the
effectiveness of the proposed Video Generation Agents (VGAs) from
two perspectives: (i) system-level comparison against conventional
production pipelines, and (ii) the contribution of different reward
components to optimization. Specifically, we assess three aspects:
generation quality, including automated and human preference
evaluations; and interest alignment score, measuring consistency
between generated videos and target user interests. See Appendix B
for details.
System-level Comparison.We compare VGAs with a conven-
tionalworkflow baseline—a hand-crafted pipeline composed of in-
struction generation, rough-cut (visual clip composition), and fine-
cut (TTS synthesis and post-production effects) stages executed
in a fixed order. Such rigid execution prevents adaptation to di-
verse user-specific generation requirements, motivating the agentic
design of VGAs.
As shown in Table 3, VGAs consistently outperform the baseline
across all metrics. The gains stem from two capabilities:reason-
ing, enabled by a hierarchically structured end-to-end framework
that supports coherent cross-modal planning; andreflection, which
improves output quality through iterative self-correction and re-
planning, capped at two iterations to maintain inference latency
comparable to the workflow baseline while achieving substantial
quality gains.
Reward Contribution Analysis.We analyze the contribution of
individual reward components under our synergistic cross-domain
rewards. Since user feedback serves as the primary objective and is
always retained, we focus the ablation on the two constraint-side
rewards: video quality and interest alignment.
As shown in Table 4, each video-quality sub-reward is evaluated
on its own dedicated test set, withBasedenoting the policy trained
without any reward optimization. Each sub-reward—visual fidelity,
audio alignment, and effect enhancement—independently improves
the Automated Win Rate over the base, and jointly optimizing
all three yields the strongest result, confirming the necessity of
balancing all three perceptual aspects.
Building on the quality rewards, incorporating the interest align-
ment reward further lifts the Interest Alignment Score (0.707 →
0.828), indicating substantially stronger consistency between gen-
erated content and user interest.Overall, these results show that the quality and alignment re-
wards play complementary roles—the former safeguards perceptual
fidelity while the latter enforces semantic relevance—and their joint
optimization, anchored by the primary user-feedback objective, pro-
duces a more robust and user-aligned video generation policy.
4.2.4 Qualitative Analysis of Personalized Video Generation.Fig-
ure 4 illustrates the end-to-end pipeline of our RaG framework,
where user interests are directly transformed into video generation.
Given a representative user profile (female, 25–34) interested in
young-mother lifestyle, baby care, household essentials, and value-
oriented shopping, the Generative Recommendation Model (GRM)
first infers Disentangled SIDs (D-SIDs) that jointly capture content
and creative preferences. These D-SIDs are mapped into structured
embeddings and, in this advertising scenario, further enriched with
the optional metadata factor 𝐷metaencoding product information
and marketing topics. Conditioned on these representations, the
Instruction Model (IM) produces shot-level production instructions,
which are subsequently executed by the Video Generation Agents
(VGAs) to coordinate visual, audio, and effect generation. The re-
sulting video achieves high quality with strong alignment to user
interests.
5 Related Work
Retrieval-Based Recommendation.Traditional recommenda-
tion systems [ 3,5,9,13,34,36] follow a retrieve-and-rank paradigm
over a fixed pool of pre-produced items.
Recent advances explore generative recommendation by model-
ing item IDs as discrete tokens and formulating recommendation as
next-token prediction over Semantic IDs [ 24,31,33]. To meet indus-
trial latency constraints, efficient architectures have been further
proposed for large-scale SID prediction [7, 27, 35].
However, these methods still rely on retrieving from a static con-
tent pool conditioned on predicted tokens, leaving recommendation
and content creation fundamentally decoupled and preventing end-
to-end optimization.
Personalized AI-Generated Content.Recent progress in gener-
ative models has motivated a shift from retrieving pre-produced
content to generating personalized content conditioned on user

Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale
preferences. Early efforts explore preference-guided LLM gener-
ation or conditioning diffusion models on user signals for image
synthesis [18, 23, 30].
These ideas have been extended to richer modalities, including
personalized advertising text generation [ 2] and dialogue-based
preference elicitation for visual content generation [ 25]. More re-
cently, NextAds [ 26] studies personalized video advertising by con-
ditioning generation on observed user preferences, but focuses
primarily on the generation module without modeling an end-to-
end pipeline from user interest representation to controllable video
production, and does not consider industrial deployment efficiency
and cost.
Overall, existing approaches mostly treat user interest modeling
and controllable content generation as separate tasks, leaving room
for a unified framework that jointly optimizes both within a closed-
loop industrial system.
6 Conclusion
We proposeRecommendation-as-Generation(RaG), a unified
paradigm that shifts recommendation toward generation-driven
personalization. RaG bridges user interest modeling and control-
lable video generation through Disentangled Semantic IDs as a
shared interface, scalable Video Generation Agents for industrial
deployment, and Synergistic Cross-Domain Reward Learning for
closed-loop optimization. Online A/B tests show that RaG con-
sistently improves ad revenue over strong commercial baselines,
demonstrating the effectiveness of the proposed paradigm in real-
world industrial settings.
RaG currently serves nearline rather than in real time, with
VGAs being the dominant latency bottleneck. Future work will
fold the Instruction Model into VGAs for a tighter generation path,
and further accelerate VGAs through stronger model distillation
and inference optimization, moving toward on-the-fly personalized
generation.
References
[1] Anthropic. Introducing Claude 4.5 Sonnet. https://www.anthropic.com/news/
claude-4-5-sonnet, September 2025. Anthropic announcement.
[2]Junyi Chen, Lu Chi, Siliang Xu, Shiwei Ran, Bingyue Peng, and Zehuan Yuan.
Hllm-creator: Hierarchical llm-based personalized creative generation.arXiv
preprint arXiv:2508.18118, 2025.
[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,
Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al.
Wide & deep learning for recommender systems. InProceedings of the 1st
workshop on deep learning for recommender systems, pages 7–10, 2016.
[4]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen
Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodal-
ity, long context, and next generation agentic capabilities.arXiv preprint
arXiv:2507.06261, 2025.
[5] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube
recommendations. InProceedings of the 10th ACM conference on recommender
systems, pages 191–198, 2016.
[6]Tri Dao. FlashAttention-2: Faster attention with better parallelism and work
partitioning. InInternational Conference on Learning Representations (ICLR), 2024.
[7] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang
Luo, and Guorui Zhou. Onerec: Unifying retrieve and rank with generative
recommender and iterative preference alignment.arXiv preprint arXiv:2502.18965,
2025.
[8]Google DeepMind. Veo 3: Our most capable video generation model. https:
//deepmind.google/models/veo/, May 2025. Google DeepMind product announce-
ment.
[9] Jian Jia, Yipei Wang, Yan Li, Honggang Chen, Xuehan Bai, Zhaocheng Liu, Jian
Liang, Quan Chen, Han Li, Peng Jiang, et al. Learn: knowledge adaptation fromlarge language model to recommendation for practical industrial application. In
Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages
11861–11869, 2025.
[10] Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu,
Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng,
et al. Gdpo: Group reward-decoupled normalization policy optimization for
multi-reward rl optimization.arXiv preprint arXiv:2601.05242, 2026.
[11] Xinchen Luo, Jiangxia Cao, Tianyu Sun, Jinkai Yu, Rui Huang, Wei Yuan, Hezheng
Lin, Yichen Zheng, Shiyao Wang, Qigen Hu, Changqing Qiu, Jiaqi Zhang,
Xu Zhang, Zhiheng Yan, Jingming Zhang, Simin Zhang, Mingxing Wen, Zhaojie
Liu, and Guorui Zhou. QARM: quantitative alignment multi-modal recommen-
dation at kuaishou. InCIKM, pages 5915–5922. ACM, 2025.
[12] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin,
Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multi-
modal embedding for videos, images, and visual documents.arXiv preprint
arXiv:2507.04590, 2025.
[13] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-
Jean Wu, Alisson G Azzolini, et al. Deep learning recommendation model for
personalization and recommendation systems.arXiv preprint arXiv:1906.00091,
2019.
[14] OpenAI. GPT-5.1 Instant and GPT-5.1 Thinking system card adden-
dum. https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_
system_card.pdf, November 2025. OpenAI technical report addendum.
[15] OpenAI. Sora 2 system card. https://cdn.openai.com/pdf/50d5973c-c4ff-4c2d-
986f-c72b5d0ff069/sora_2_system_card.pdf, 2025.
[16] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan,
Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al.
Recommender systems with generative retrieval.Advances in Neural Information
Processing Systems, 36:10299–10315, 2023.
[17] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen,
Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, et al. Seedance 1.5
pro: A native audio-visual joint generation foundation model.arXiv preprint
arXiv:2512.13507, 2025.
[18] Xiaoteng Shen, Rui Zhang, Xiaoyan Zhao, Jieming Zhu, and Xi Xiao. Pmg:
Personalized multimodal generation with large language models. InProceedings
of the ACM Web Conference 2024, pages 3833–3843, 2024.
[19] Adam Stooke, Joshua Achiam, and Pieter Abbeel. Responsive safety in reinforce-
ment learning by PID lagrangian methods. In Hal Daumé III and Aarti Singh,
editors,Proceedings of the 37th International Conference on Machine Learning,
volume 119 ofProceedings of Machine Learning Research, pages 9133–9143. PMLR,
13–18 Jul 2020. URL https://proceedings.mlr.press/v119/stooke20a.html.
[20] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai,
Sainan Guo, Feng Han, Jingbin He, Kang He, et al. Kling-omni technical report.
arXiv preprint arXiv:2512.16776, 2025.
[21] Qwen Team. Qwen2.5-vl technical report.arXiv preprint arXiv:2502.13923, 2025.
[22] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen,
Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng
Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu
Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu,
Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang,
Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng
Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming
Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya
Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao
Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and
Ziyu Liu. Wan: Open and advanced large-scale video generative models.arXiv
preprint arXiv:2503.20314, 2025.
[23] Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, and Tat-Seng Chua. Genera-
tive recommendation: Towards next-generation recommender paradigm.arXiv
preprint arXiv:2304.03516, 2023.
[24] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-
Kiong Ng, and Tat-Seng Chua. Learnable item tokenization for generative
recommendation. InProceedings of the 33rd ACM International Conference on
Information and Knowledge Management, pages 2400–2409, 2024.
[25] Xianquan Wang, Zhaocheng Du, Huibo Xu, Shukang Yin, Yupeng Han, Jieming
Zhu, Kai Zhang, and Qi Liu. Personalized visual content generation in conver-
sational systems. InProceedings of the 39th Conference on Neural Information
Processing Systems (NeurIPS 2025), 2025.
[26] Yiyan Xu, Ruoxuan Xia, Wuqiang Zheng, Fengbin Zhu, Wenjie Wang, and Fuli
Feng. Nextads: Towards next-generation personalized video advertising.arXiv
preprint arXiv:2603.02137, 2026.
[27] Ben Xue, Dan Liu, Lixiang Wang, Mingjie Sun, Peng Wang, Pengfei Zhang,
Shaoyun Shi, Tianyu Xu, Yunhao Sha, Zhiqiang Liu, et al. Generative recom-
mendation for large-scale advertising.arXiv preprint arXiv:2602.22732, 2026.
[28] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical
report.arXiv preprint arXiv:2412.15115, 2024.

Yanhua Cheng et al.
[29] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025.
[30] Hao Yang, Jianxin Yuan, Shuai Yang, Linhe Xu, Shuo Yuan, and Yifan Zeng. A new
creative generation pipeline for click-through rate with stable diffusion model.
InCompanion Proceedings of the ACM Web Conference 2024, pages 180–189, 2024.
[31] Jun Yin, Zhengxin Zeng, Mingzheng Li, Hao Yan, Chaozhuo Li, Weihao Han,
Jianjin Zhang, Ruochen Liu, Hao Sun, Weiwei Deng, et al. Unleash llms potential
for sequential recommendation by coordinating dual dynamic index mechanism.
InProceedings of the ACM on Web Conference 2025, pages 216–227, 2025.
[32] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong
Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. Qwen3 embedding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176, 2025.
[33] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen,
and Ji-Rong Wen. Adapting large language models by integrating collaborative
semantics for recommendation. In2024 IEEE 40th International Conference on
Data Engineering (ICDE), pages 1435–1448. IEEE, 2024.
[34] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui
Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through
rate prediction. InProceedings of the 24th ACM SIGKDD international conference
on knowledge discovery & data mining, pages 1059–1068, 2018.
[35] Guorui Zhou, Hengrui Hu, Hongtao Cheng, Huanjie Wang, Jiaxin Deng, Jinghao
Zhang, Kuo Cai, Lejian Ren, Lu Ren, Liao Yu, Pengfei Zheng, Qiang Luo, Qianqian
Wang, Qigen Hu, Rui Huang, Ruiming Tang, Shiyao Wang, Shujie Yang, Tao
Wu, Wuchao Li, Xinchen Luo, Xingmei Wang, Yi Su, Yunfan Wu, Zexuan Cheng,
Zhanyu Liu, Zixing Zhang, Bin Zhang, Boxuan Wang, Chaoyi Ma, Chengru
Song, Chenhui Wang, Chenglong Chu, Di Wang, Dongxue Meng, Dunju Zang,
Fan Yang, Fangyu Zhang, Feng Jiang, Fuxing Zhang, Gang Wang, Guowang
Zhang, Han Li, Honghui Bao, Hongyang Cao, Jiaming Huang, Jiapeng Chen,
Jiaqiang Liu, Jinghui Jia, Kun Gai, Lantao Hu, Liang Zeng, Qiang Wang, Qidong
Zhou, Rongzhou Zhang, Shengzhe Wang, Shihui He, Shuang Yang, Siyang Mao,
Sui Huang, Tiantian He, Tingting Gao, Wei Yuan, Xiao Liang, Xiaoxiao Xu,
Xugang Liu, Yan Wang, Yang Zhou, Yi Wang, Yiwu Liu, Yue Song, Yufei Zhang,
Yunfeng Zhao, Zhixin Ling, and Ziming Li. Onerec-v2 technical report, 2025.
URL https://arxiv.org/abs/2508.20900.
[36] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai.
Learning tree-based deep model for recommender systems. InProceedings of
the 24th ACM SIGKDD international conference on knowledge discovery & data
mining, pages 1079–1088, 2018.

Recommendation as Generation: Unifying Personalized Video Generation and Recommendation at Industrial Scale
Table 5:Inference efficiency comparison across modules of RaG
system.
ComponentD-SIDs
(Nearline)GRM
(Online)IM
(Nearline)VGAs
(Nearline)
Latency∼4s∼100ms∼2.5s∼180s
A Runtime Analysis of RaG Modules
We deploy the proposed RaG system in Kuaishou’s large-scale adver-
tising and recommendation infrastructure, serving over 400 million
users. To better understand system efficiency, we conduct a runtime
analysis of each core component under both online and nearline
deployment settings. Specifically, the Generative Recommendation
Model (GRM) operates in an online serving regime for real-time
recommendation, while Disentangled Semantic IDs (D-SIDs), the
Instruction Model (IM), and the Video Generation Agents (VGAs)
are executed in a nearline pipeline due to their higher computa-
tional cost and generation latency. All latency numbers reported in
Table 5 are measured under live production traffic, reflecting the
actual serving conditions.
B Evaluation Metrics for Video Quality and
Interest Alignment
To provide a fair and unbiased evaluation of the proposed Video
Generation Agents (VGAs), we adopt an evaluation protocol that
is fully decoupled from the reward functions used during training.
Specifically, we assess generated videos from three complementary
perspectives:instruction-level interest alignment score,automated
multi-dimensional quality evaluation, andhuman preference assess-
ment. All evaluation scores are computed using external judges or
human annotators, ensuring that the reported results reflect gener-
alization quality rather than optimization-specific reward fitting.
Interest Alignment Score.We first evaluate whether generated
videos faithfully follow their corresponding video production in-
structions derived from interest SIDs. To mitigate single-judge
bias, we employ an ensemble of three state-of-the-art multimodal
evaluators—GPT-5.1 [ 14], Gemini-2.5 Pro [ 4], and Claude-4.5 Son-
net [ 1]—each independently scoring the same benchmark of 1,000
multi-category video instances under the identical protocol defined
in Box B.1. Each generated video is rated along five dimensions—
semantic consistency, attribute accuracy, thematic alignment, com-
pleteness, and narrative coherence—producing a continuous align-
ment score in[0,1]. We report the per-instance average across the
three judges as the final Interest Alignment Score.
Automated Quality Evaluation.Beyond semantic alignment,
we further evaluate the overall production quality of generated
videos using the same three-judge ensemble—GPT-5.1 [ 14], Gemini-
2.5 Pro [ 4], and Claude-4.5 Sonnet [ 1]—under the identical protocol
defined in Box B.2. Each judge independently rates every video
along four aspects: (1) instruction attractiveness, measuring hook
quality, pacing, and call-to-action effectiveness; (2) BGM compati-
bility, evaluating music-tone consistency and beat synchronization;
(3) SFX and sticker design quality, assessing visual effects and sub-
title design; and (4) instruction–visual alignment, measuring the
consistency between visual progression and instruction semantics.Per-instance scores are averaged across the three judges, yield-
ing two metrics: a normalizedAutomated Scorein [0,1]and an
Automated Win Rateunder the Good-Same-Bad (GSB) setting.
Human Preference Assessment.To further validate real-world
perceptual quality and user preference alignment, we conduct a
human evaluation study with 20 annotators from diverse profes-
sional backgrounds—including algorithm engineers, product man-
agers, and advertising clients—covering both algorithm-side and
business-side perspectives to reduce single-role bias. Each annota-
tor performs 50 pairwise comparisons between generated videos
and baseline results under a Good-Same-Bad (GSB) protocol, yield-
ing 1,000 pairwise judgments in total. All comparisons are presented
in a blind, randomized order, with each video pair independently
evaluated by at least three annotators to mitigate individual sub-
jectivity; we report the majority-vote outcome as theUser Study
Win Rate.
Box B.1: Interest Alignment Prompt
Role
You are an expert evaluator assessing the alignment between the
video production instructions and the corresponding generated
video. Focus only on semantic and creative consistency, while
ignoring production quality (e.g., resolution, smoothness, or
visual artifacts).
Inputs
•Instructions: {instructions}
•Video: {video}
Task
Evaluate the video along the following five dimensions, each scored
in[0,1].
[A1] Content Fidelity
•Subjects, actions, and scenes match the instruction.
[A2] Attribute Accuracy
•Visual attributes and spatial-temporal relationships are
correctly represented.
[A3] Intent & Theme Alignment
•Creative intent, mood, and stylistic cues align with the
instruction.
[A4] Completeness
•All key elements are included without hallucinated content.
[A5] Narrative Coherence
•The temporal progression and story flow remain coherent.
Scoring Scale
•0.9–1.0: Perfect alignment
•0.7–0.9: Minor deviations
•0.5–0.7: Moderate deviations
•0.3–0.5: Major missing elements
•0.0–0.3: Largely unrelated
Output Format
{
"content_fidelity": {"score": 0.0},
"attribute_accuracy": {"score": 0.0},
"intent_theme_alignment": {"score": 0.0},
"completeness": {"score": 0.0},
"narrative_coherence": {"score": 0.0},
"overall_alignment_score": 0.0
}
Rules
Scores should be continuous in[ 0,1]. The overall score is a
holistic judgment rather than the arithmetic mean of sub-scores.
Do not consider production quality unless explicitly required by
the instruction.

Yanhua Cheng et al.
Box B.2: Video Quality Assessment Prompt
Role
You are a professional short-video advertising evaluator assessing
video quality from editing and audio-visual perspectives.
Task
Evaluate the video across four dimensions with a total score of
100 points.
[D1] Instruction Attractiveness (25)
•Hook quality: pain-point, suspense, benefit-first, or contrast
design.
•Pacing and structure: logical progression without redundant
segments.
•CTA effectiveness: clarity, urgency, and consistency with
opening intent.
[D2] BGM Compatibility (25)
•Mood and tempo alignment with video content.
•Synchronization between cuts and music beats.
•Balanced audio volume and speech clarity.
[D3] SFX & Sticker Design (25)
•Effectiveness of transition, emphasis, ambient, and emotional
SFX.
•Consistency and readability of subtitles, tags, arrows, and
motion effects.
[D4] Instruction–Visual Alignment (25)
•Consistency between instruction keywords and visual content.
•Narrative flow and temporal coherence.
•Absence of visual-information gaps or dead-air segments.
Output Format
{
"instruction_attractiveness": {"total": 0},
"bgm_compatibility": {"total": 0},
"sfx_sticker_design": {"total": 0},
"instruction_visual_alignment": {"total": 0},
"final_summary": {
"total_score": 0,
"grade": "S/A/B/C/D"
}
}
Rules
The total score ranges from 0 to 100. Grades are defined as: S (90+),
A (75+), B (60+), C (45+), and D (<45). All evaluations should
be evidence-based and supported by specific visual or temporal
observations.
C Details of Generative Recommendation
Model
Our Generative Recommendation Model (GRM) follows an archi-
tecture similar to GR4AD [27].
In training, the model takes two inputs: (1) user contextC, con-
sisting of static profile features Fprof(e.g., age, gender, region, device
type) and multi-granularity behavior sequences Fseq, where each
interaction is encoded via a sparse embedding table into latent inter-
est tokens; and (2) prefix SID sequence (𝐵𝑂𝑆,𝑠1
content,𝑠2
content,𝑠1
creative,
𝑠2
creative)derived from the target item. We formulate GRM as an
autoregressive sequence modeling problem, predicting the full SID
sequence conditioned on the prefix, where the generated SIDs serve
as a discrete representation of user interests, optimized via token-
level cross-entropy loss, followed by reinforcement learning fine-
tuning with constrained GDPO to further improve user feedback
and interest alignment.
Concretely, each SID token is retrieved from a sparse embed-
ding table and projected into a 768-dimensional latent space, then
processed by a 7-layer Transformer decoder (LazyDecoder) with
hidden size 768, FFN size 3072, 12 attention heads, and vocabulary
size 8192. FlashAttention [ 6] is adopted for efficient computation.We train the model on 8 GPUs with batch size 8192 using Adam
(𝑙𝑟= 1×10−4). During inference, beam search (beam size 512) is
used, achieving 130 QPS throughput.