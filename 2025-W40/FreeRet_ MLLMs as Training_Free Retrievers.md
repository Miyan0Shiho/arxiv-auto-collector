# FreeRet: MLLMs as Training-Free Retrievers

**Authors**: Yuhan Zhu, Xiangyu Zeng, Chenting Wang, Xinhao Li, Yicheng Xu, Ziang Yan, Yi Wang, Limin Wang

**Published**: 2025-09-29 11:28:42

**PDF URL**: [http://arxiv.org/pdf/2509.24621v1](http://arxiv.org/pdf/2509.24621v1)

## Abstract
Multimodal large language models (MLLMs) are emerging as versatile
foundations for mixed-modality retrieval. Yet, they often require heavy
post-hoc training to convert them into contrastive encoders for retrieval. This
work asks: Can off-the-shelf MLLMs serve as powerful retrievers without
additional training? We present FreeRet, a plug-and-play framework that turns
any MLLM into a two-stage retriever. FreeRet first derives semantically
grounded embeddings directly from the model for fast candidate search, and then
exploits its reasoning ability for precise reranking. The framework contributes
three advances: bypassing lexical alignment layers to obtain semantically
faithful embeddings, conditioning representation generation with explicit
priors, and mitigating framing effect in reranking via neutral choice framing.
On the MMEB and MMEB-V2 benchmarks spanning 46 datasets, FreeRet substantially
outperforms models trained on millions of pairs. Beyond benchmarks, FreeRet is
model-agnostic and scales seamlessly across MLLM families and sizes, preserves
their generative abilities, supports arbitrary modality combinations, and
unifies retrieval, reranking, and generation into end-to-end RAG within a
single model. Our findings demonstrate that pretrained MLLMs, when carefully
harnessed, can serve as strong retrieval engines without training, closing a
critical gap in their role as generalists.

## Full Text


<!-- PDF content starts -->

Preprint
FREERET: MLLMS ASTRAINING-FREERETRIEVERS
Yuhan Zhu1,2Xiangyu Zeng1,2Chenting Wang2,3Xinhao Li1
Yicheng Xu2,4Ziang Yan2,5Yi Wang2Limin Wang1,2∗
1Nanjing University2Shanghai AI Laboratory
3Shanghai Jiaotong University4Institute of Science Tokyo5Zhejiang University
yuhan.zhu@smail.nju.edu.cn lmwang@nju.edu.cn
ABSTRACT
Multimodal large language models (MLLMs) are emerging as versatile founda-
tions for mixed-modality retrieval. Yet, they often require heavy post-hoc training
to convert them into contrastive encoders for retrieval. This work asks:Can off-
the-shelf MLLMs serve as powerful retrievers without additional training?We
presentFreeRet, a plug-and-play framework that turns any MLLM into a two-
stage retriever. FreeRet first derives semantically grounded embeddings directly
from the model for fast candidate search, and then exploits its reasoning ability for
precise reranking. The framework contributes three advances: bypassing lexical
alignment layers to obtain semantically faithful embeddings, conditioning repre-
sentation generation with explicit priors, and mitigating framing effect in rerank-
ing via neutral choice framing. On the MMEB and MMEB-V2 benchmarks span-
ning 46 datasets, FreeRet substantially outperforms models trained on millions
of pairs. Beyond benchmarks, FreeRet is model-agnostic and scales seamlessly
across MLLM families and sizes, preserves their generative abilities, supports ar-
bitrary modality combinations, and unifies retrieval, reranking, and generation
into end-to-end RAG within a single model. Our findings demonstrate that pre-
trained MLLMs, when carefully harnessed, can serve as strong retrieval engines
without training, closing a critical gap in their role as generalists.
1 INTRODUCTION
Multimodal retrieval, the process of retrieving relevant items across multiple modalities, is a cor-
nerstone of modern AI systems. It underlies applications ranging from web search (Mitra et al.,
2017; Huang et al., 2020) and retrieval-augmented generation (RAG) (Lewis et al., 2020; Gao et al.,
2023), to embodied agents (Singh et al., 2025) and personalized recommendation (Rajput et al.,
2023). Conventional solutions rely on two stages: embedding-based candidate search followed
by reranking for accuracy. CLIP-style dual encoders (Radford et al., 2021; Li et al., 2022) have
long been the workhorse of this paradigm, but they exhibit fundamental limitations: they struggle
with long queries, compositional semantics, and interleaved multimodal inputs. These shortcomings
highlight the need for a more generalizable foundation.
Multimodal large language models (MLLMs) offer such a foundation. With powerful reasoning
and flexible input handling, they promise to unify understanding across modalities. Yet most recent
efforts adapt MLLMs to retrieval through heavy post-hoc fine-tuning (Fig. 1(a)) (Lin et al., 2024;
Zhang et al., 2024b; Liu et al., 2025b). This paradigm, however, encounters two persistent obstacles.
First, it demands massive paired data and expensive fine-tuning for every new backbone or modality
configuration, hampering scalability. Second, its generalization remains fragile: without in-domain
supervision, even large, carefully curated models often perform poorly on standard benchmarks.
Can we instead harness MLLMs for retrievalwithouttraining, letting them act simultaneously as
embedders and rerankers?Early training-free attempts embed generated token states (Jiang et al.,
2024a; 2023), but these representations are coarse and generally omit reranking, a critical stage for
robust performance. Other approaches integrate learned rerankers, but they sacrifice efficiency and
modularity by requiring additional supervision and model components.
∗Corresponding author
1arXiv:2509.24621v1  [cs.CV]  29 Sep 2025

Preprint
Embedding 
expert
Pre-trained 
MLLMRanking 
expert
Pre-trained 
MLLM(a) Previous  post-training retrievers
(a) FreeRet : Training -free MLLM retrieval
 (c) Performance on the MMEB BenchmarkExpensive multimodal 
data curationHeavy post -hoc
fine-tuning
Initial pool
Top-kcandidates
Embedding -based Searching
MCQ -based Reranking
Final rankingFreeRet
Figure 1:Comparison between prior post-training retrievers and our FreeRet.(a) Existing
methods rely on extensive data curation and costly fine-tuning to constructseparateembedding
and reranking modules. (b) FreeRet directly employs MLLMs asunifiedembedders and rerankers
without any extra training. (c) On the MMEB benchmark covering 36 datasets, FreeRet outperforms
models trained on millions of pairs and matches the best methods supervised directly on MMEB.
To address this gap, we proposeFreeRet, a plug-and-play framework that transforms any off-the-
shelf MLLM into a competitive two-stage retriever (Fig. 1(b)). FreeRet first extracts embeddings
for efficient candidate search, then prompts the same model to conduct fine-grained reranking. Cru-
cially, it requires no parameter updates, auxiliary models, or external data. By fully exploiting both
the representational and reasoning capacities of MLLMs, FreeRet demonstrates thattraining-free
retrieval is not only feasible but also state-of-the-art competitive.
Our framework rests on three key contributions: (1) We refine embedding quality by bypassing the
final MLP before the LM head, which enforces surface-level lexical alignment at the expense of
semantic structure. Removing it yields embeddings that better capture underlying meaning. (2) We
stabilize the embedding space via controlled summarization prompts that inject semantic, denois-
ing, and contextual priors. This improves semantic focus and task relevance. (3) We uncover an
LLMframing effectin reranking: logically equivalent label formats (e.g.Yes/No vs. True/False)
yield divergent accuracies due to pretraining biases (Zhao et al., 2021). We mitigate this by casting
reranking as multiple-choice questions (MCQ), which elicit more neutral and reasoned judgments.
Evaluated on MMEB (Jiang et al., 2024b), a comprehensive suite of 36 datasets across four
meta-tasks, FreeRet consistently delivers strong gains (Fig. 1(c)). Notably, FreeRet-2B outper-
forms GME-7B (Zhang et al., 2024b), trained on 8M multimodal pairs, and dramatically surpasses
MMRet-7B (Zhou et al., 2024), trained on 26.2M pairs. Remarkably, FreeRet-7B competes head-to-
head with methods explicitly supervised on MMEB, while on the video subset of MMEB-V2 (Meng
et al., 2025), it exceeds even post-trained approaches by large margins. Together, these results high-
light the strength of a purely training-free paradigm.
Beyond empirical gains, FreeRet delivers broader implications. It eliminates the costly adaptation
barrier, allowing immediate deployment of any MLLM across scales and architectures (Tab. 1).
It naturally supports arbitrary modality combinations when applied to omni-modal models like
Qwen2.5-Omni (Xu et al., 2025a) (Fig. 6). By avoiding fine-tuning, FreeRet fully retains the
instruction-following, conversational, and reasoning capacity of pretrained MLLMs. Moreover, it
streamlines the RAG framework, unifying retrieval, reranking, and generation within a single model,
and enhances long video understanding by retrieving relevant clips before deeper reasoning (Tab. 4).
Additionally, FreeRet serves as a diagnostic lens, extending the evaluation of MLLMs beyond con-
ventional QA settings to include retrieval-oriented tasks.
In summary, FreeRet establishes pretrained MLLMs as competitive, versatile, and training-free re-
trieval systems. It challenges the necessity of large-scale supervised adaptation and points toward a
future where a single generalist model unifies retrieval with broader multimodal understanding.
2

Preprint
2 RELATEDWORK
Multimodal Large Language Models (MLLMs).The evolution of MLLMs reflects a steady
progression from alignment to generation. CLIP (Radford et al., 2021) demonstrated the power
of contrastive pretraining on web-scale image–text pairs, yielding highly transferable embeddings.
BLIP (Li et al., 2022) advanced this framework by unifying contrastive and generative objectives.
Flamingo (Alayrac et al., 2022) introduced gated cross-attention for few-shot multimodal learning,
while LLaV A (Liu et al., 2023; Li et al., 2024) pushed instruction tuning into visual dialogue to
enhance interactive reasoning. More recent efforts, including GPT-4V (Achiam et al., 2023), Qwen-
VL (Wang et al., 2024a; Bai et al., 2025), and InternVL (Chen et al., 2024; Zhu et al., 2025),
scale these ideas toward general-purpose multimodality, addressing tasks from grounded reasoning
to document-level understanding. Collectively, these advances trace a trajectory from multimodal
alignment to complex reasoning, establishing the backbones for multimodal retrieval.
Training-based Retrieval with MLLMs.Early multimodal retrieval has largely relied on CLIP-
style dual encoders (Radford et al., 2021; Zhai et al., 2023; Sun et al., 2023), yet this paradigm
faces two core challenges. First, text encoders struggle with long or compositional inputs that de-
mand fine-grained reasoning. Second, the unimodal design prevents robust handling of interleaved
content, limiting applications such as composed image retrieval (V o et al., 2019). To address these
shortcomings, recent work repurposes MLLMs as retrieval backbones, treating MLLMs as universal
encoders, and fine-tunes them with contrastive learning. Within this framework, researchers have
explored a spectrum of techniques, ranging from hard negative mining (Gu et al., 2025; Schneider
et al., 2025; Lan et al., 2025; Xue et al., 2025) and modality-aware sampling (Lyu et al., 2025; Kong
et al., 2025) to reinforcement learning (Zhao et al., 2025) and joint optimization with generative
objectives (Ouali et al., 2025; Yu et al., 2025). Parallel lines of work focus on curating large-scale
mixed-modality corpora (Zhang et al., 2024b; Zhou et al., 2024; Chen et al., 2025; Liu et al., 2025a),
designing broad evaluation suites (Lin et al., 2024; Jiang et al., 2024b; Xiao et al., 2025), train-
ing rerankers (Xu et al., 2025b), and building end-to-end pipelines that combine embedding with
reranking (Lin et al., 2024; Liu et al., 2025b; Cui et al., 2025). Despite these advances, training-
based approaches retain fundamental bottlenecks. They demand massive multimodal datasets and
expensive re-training whenever a new backbone or modality configuration is introduced. More im-
portantly, models optimized for one benchmark (e.g.MMEB) often transfer poorly to others (e.g.
MIEB), exposing weak generalization without benchmark-specific supervision.
Training-free Retrieval with Auto-Regressive Models.Most attempts at training-free retrieval
with language models have mostly focused on the text-only setting. They extract embeddings from
internal hidden states without further optimization. Early approaches such as PromptEOL (Jiang
et al., 2023) design handcrafted prompts (e.g., “The sentence means in one word:”) so that the
hidden state of the next token approximates a sentence-level representation. MetaEOL (Lei et al.,
2024) and GenEOL (Thirukovalluru & Dhingra, 2024) improve robustness by combining multiple
embeddings, while Zhang et al. (2024a) enhances expressivity using chain-of-thought prompting or
external knowledge injection. Token-Prepend (Fu et al., 2024) and Echo-Embedding (Springer et al.,
2024) mitigate the causal attention bias that suppresses early tokens, while MoEE (Li & Zhou, 2024)
fuses routing weights with hidden states in MoE architectures to yield more expressive embeddings.
In contrast, the multimodal setting remains underexplored (Jiang et al., 2024a; Ju & Lee, 2025).
They often produce coarse representations that capture only partial cross-modal alignment, and they
lack integration with reranking mechanisms which is crucial to accurate retrieval. As a result, their
performance lags far behind approaches that permit task-specific training.
3 FREERET
3.1 PRELIMINARY: TRAINING-FREEEMBEDDINGEXTRACTION
A representative attempt of MLLMs training-free embedding is E5-V (Jiang et al., 2024a). Given
an inputxcomposed of arbitrary modality combinations, E5-V applies a fixed prompting template:
“[x]\n Summary above content in one word:”,(1)
and queries the MLLM to generate a single tokeny. Leth L(y)denote the hidden state ofyat the
final transformer layer indexL. The embedding of the inputxis then defined ase(x) =h L(y).
3

Preprint
(a) Cosine similarity between adjacent layer hidden
states and their alignment with LM head.
(b) Layer-wise hidden state similarity for 250 syn-
onym pairs (mean ± standard deviation).
Figure 2:Probing experiments on lexicalization pressure. Results for 3B and 32B variants are
provided in Fig. 7.
This strategy preserves all parameters and thereby inherits the reasoning and generalization abilities
from training. Yet its effectiveness as a generic embedder is constrained. The hidden stateh L(y)
is optimized for predicting the next token rather than for encoding input semantics, which biases it
toward surface-level lexical statistics (Li & Zhou, 2024). Furthermore, current training-free designs
often ignore the reranking stage, which is crucial for accurate retrieval.
To overcome these issues. In § 3.2, we examine how the lexicalization pressure induced by the
final MLP layer limits the semantic capacity of hidden states. In § 3.3, we move beyond one-token
summarization and design a controlled generation objective. Finally, in § 3.4, we investigate the role
of MLLMs as rerankers and identify a framing-effect phenomenon that undermines robustness.
3.2 LEXICALIZATIONPRESSURE INMLLM REPRESENTATIONS
As discussed in § 3.1, the final hidden stateh L(y)is optimized for generating vocabulary logits
rather than preserving semantic structure, leading to suboptimal embedding quality. Prior studies
indicate that internal representations evolve across depth: early and intermediate layers capture
semantic abstractions, while later layers reshape these abstractions toward task-specific objectives
(Zeiler & Fergus, 2014). In MLLMs, we refer to this final transformation aslexicalization pressure:
the process by which semantic features are projected into a space for discrete lexical prediction.
Probing Experiment.We begin by examining where lexicalization pressure arises within the
model. Using Qwen2.5-VL at three scales (3B, 7B, and 32B), we probe the last five transformer
layers. Each layer consists of an attention sub-layer and an MLP sub-layer, and we denote their
outputs ashAttn
ℓandhMLP
ℓfor layer indexℓ. We assess representational shifts by the cosine similarity
between consecutive sub-layer outputs:αAttn
ℓ= cos 
hMLP
ℓ−1, hAttn
ℓ
,αMLP
ℓ= cos 
hAttn
ℓ, hMLP
ℓ
. A
lowerαindicates a stronger distortion of representations. Next, we measure how strongly each hid-
den state is pulled into the lexical prediction space. LetW∈Rd×|V|be the LM head and letw y∗
be the column for predicted tokeny∗, we defineβAttn
ℓ= cos 
hAttn
ℓ,wy∗
,βMLP
ℓ= cos 
hMLP
ℓ,wy∗
.
Here, a higherβcorresponds to stronger alignment with the lexical head. The results in Fig. 2a
reveal consistent trends: 1)αremains very high (over 0.9) across most layers but drops sharply after
the final MLP (below 0.3); 2)βstays low (around 0) across earlier layers but rises abruptly right after
the last MLP (up to 0.5). These together point tothe final MLP as the focal point of lexicalization,
transforming semantically rich intermediate states into vectors aligned with token prediction.
Effect on Semantic Retention.As shown in Fig. 2b, cosine similarity remains around 94% across
most layers, but declines to 87% after the final MLP. This suggests that lexicalization pressure
compels hidden states to converge on coordinates tied to individual lexical items, erasing part of
their semantic continuity. Such embeddings are therefore less suitable for retrieval tasks that require
fine-grained semantic discrimination.
4

Preprint
On Alessia Cara’s second album,
“The Pains of growing,’’ she presents herself 
as a woman who’s still figuring herself out.
This topic of news relates to ‘music’.
Summarize the in one word .
 Summarize the             in one word .
You are required to assess if the 
news is related to the topic.
Based on Q , output one word:
• Capture the semantics of the 
image and news. Favor the topic of 
the news.
• Do not use function words, 
prepositions, or symbols.
Query (text+image )
 FreeRet  (ours)
Target (text)Query
Query Target
You are required to assess if the 
news is related to the topic.
Based on sente   , output one word:
•Capture the semantics of 
the sentence. Favor the topic of 
the sentence.
• Do not use function words,
prepositions, or symbols.Target
E5-V
Figure 3:Word-level probability visualizationfor the output “One Word” of different methods.
The top-left panel shows the input example (from N24News (Wang et al., 2021)).
Remedy.Building on these findings, we propose a simple yet effective fix: discard the final MLP
layer when producing embeddings. This choice retains the high-level abstractions encoded in deeper
layers while avoiding the distortion caused by lexicalization pressure. As a result, the final represen-
tations capture semantic content more faithfully and exhibit improved robustness in retrieval tasks.
3.3 FROMFREE-FORMSUMMARIES TOCONTROLLEDGENERATION
Early multimodal embedders (e.g.E5-V) often relied on simple prompting strategies such as
“Summarize the input in one word”. Although superficially elegant, this approach
leaves the generation process largely unconstrained and introduces several issues. First,seman-
tic loss: compressing complex multimodal signals into a single token frequently leads to overly
abstractive concepts. Second,vocabulary noise: high-frequency but uninformative words, including
articles and prepositions, pollute the embedding space. Third,weak task relevance: task-agnostic
prompts yield representations poorly aligned with specific retrieval needs.
Fig. 3 (left) illustrates these effects. E5-V often predicts vague words such as “Self” or “Searching”,
or produces spurious function words, or drifts toward semantic-related but task-irrelevant concepts
like “Growing”. Such outputs dilute semantic precision and lead to degraded retrieval performance.
We address these limitations by reframing free-form one-word summarization as acontrolled gen-
erationproblem. Crucially, our method requires neither architectural changes nor extra training; the
improvements stem purely from prompt design. We introduce three lightweight constraints:
1)Task alignment:stear the generation process toward specific task priors (“You are
reqired to assess if <placeholder> is related to <placeholder>.”).
2)Semantic grounding:anchor the summary to the input concent (“Capture the
semantics of <placeholder>”).
3)Noise suppression:eliminate trivial tokens (“Do not use function words,
prepositions, or symbols”).
This strategy preserves the simplicity of single-word outputs yet enforces structural discipline. As
shown in Fig. 3 (right), the resulting vocabularies ofQueryandTargetare more semantically aligned
with each other. Consequently, the embeddings can converge more reliably, yielding representations
that remain faithful to the original content while being better tailored to the user’s specific intent.
3.4 RERANKING WITHMLLMS: THEFRAMINGEFFECT
To refine retrieval quality, we repurpose MLLMs as point-wise rerankers (Burges et al., 2005). The
standard approach is straightforward: given a query-candidate pair, the model is asked to judge
whether they are relevant. This forms reranking as a binary classification problem, a design choice
that has become widely adopted (Zhang et al., 2025; Lin et al., 2024; Liu et al., 2025b).
Yet our study reveals that such paradigm hides surprising brittleness. The very act of framing the
binary decision, even when the logical meaning remains identical, leads to strikingly different accu-
racies. For instance, in Fig. 4, when prompting the model withYes/No,True/False, orRight/Wrong.
5

Preprint
Context -free
 instructionOutput word 
probabilities
Reply only with ‘Right’ or ‘Wrong' ‘Right’  (24.46% ) ‘Wrong’  (75.20% )
Reply only with ‘Yes' or ‘No' ‘Yes’ (33.21% )‘No’ (66.25% )
Reply only with ‘True' or ‘False' ‘True’  (49.51% )‘False’  (49.66% )
Reply only with ‘A' or ‘B' ‘A’ (49.20% ) ‘B’ (50.71% )Context -based evaluation (ImageNet -1K)
Shared prefix : “<query image> Does the 
<target object> appear in the image?”
Answer the question only with ‘Right’ or ‘Wrong’.
A. The object is clearly present in the image. 
B. The object is not present in the image.Answer the question only with ‘Yes’ or ‘No’.
Answer the question only with ‘True’ or ‘False’.
70.8%
72.5%
75.8%
80.9%Accuracy
Figure 4:LLM framing effecton benchmark accuracy (left) and inherent lexical biases in context-
free response modes (right).
One would expect these to be interchangeable, since each simply encodes a positive/negative deci-
sion. However, the model achieved 5.0% lower accuracy withRight/Wrongthan withTrue/False.
What drives this sensitivity? We posit it stems from imbalances inherited from pretraining corpora.
Words differ not only in logical role but also in social and pragmatic connotations:e.g. Yesoften
signals politeness,Noconveys refusal, andRight/Wrongcarries a moral or judgmental tone. Such
contexts may bring unintended biases to their logical use. To probe this, following “context-free in-
struction” setup in Zhao et al. (2021), the model is prompted to choose between label pairs without
any context. Ideally output logits should be uniformly distributed, but we find clear asymmetries:
pairs likeRight/WrongorYes/Noshow obvious skew, whileTrue/Falseremains closer to balance.
Intriguingly, greater bias correlates with lower downstream accuracy. This mirrors the classicfram-
ing effectin cognitive science, where equivalent choices elicit different judgments depending on
presentation. We term the analogous phenomenon here theLLM framing effect.
To mitigate this effect, we frame the ranking problem as a multiple-choice question (MCQ). Given
a query-candidate pair, the model is asked:
Task: Determine whether the candidate matches the query.
Query: {Query}
Candidate: {Candidate}
A. Yes, the candidate fully matches the query.
B. No, the candidate does not match or only partially matches.
The relevance score is then computed asSoftMax(p(‘A’))from the LM head. MCQ content may
be adjusted to suit the needs of each task. This design offers two benefits. First, it neutralizes seman-
tic and affective biases from lexical choices. Second, it mirrors multiple-choice formats prevalent
in pretraining data, enabling more stable predictions. Despite being semantically equivalent, this
simple reframing is highly effective: as shown in Fig. 4, it outperforms the commonly usedYes/No
setup by 8.4%, underscoring the effectiveness of our design.
4 EXPERIMENTS
4.1 EVALUATIONSETUP
We evaluate FreeRet on MMEB (Jiang et al., 2024b) and the video subset of MMEB-V2 (Meng
et al., 2025), spanning diverse multimodal tasks: image classification (10 datasets), visual question
answering (10), information retrieval (12), visual grounding (4), video classification (5), and video
retrieval (5). Following prior work, we report Precision@1 for all image and video tasks.
To explore generality across model families and scales, we deploy FreeRet with the Qwen se-
ries (Qwen2-VL (Wang et al., 2024a), 2B/7B, Qwen2.5-VL (Bai et al., 2025), 3B/7B/32B, and
Qwen2.5-Omni (Xu et al., 2025a)), InternVL (InternVL3 (Zhu et al., 2025), 2B/8B/14B), and
LLaV A (LLaV A-OV-7B (Li et al., 2024), LLaV A-OV-1.5-8B (Contributors, 2025)).
Baselines.On MMEB, we compare against three groups: (1) models trained in part on the MMEB-
train split (20 datasets, 662K pairs); (2) models trained on data without MMEB-train; and (3)
training-free methods. For the latter, we reproduce a training-free version of E5-V with Qwen2.5-VL
for fair comparison. On MMEB-V2, none of the baselines are directly supervised on the benchmark.
6

Preprint
Table 1:Performance comparison on the MMEB benchmark(Jiang et al., 2024b). We report
Precision@1 for all models. †: we reproduce the training-free version of E5-V .
Model MLLM Train Data (M) Classification VQA Retrieval Grounding Average
Explicitly Supervised by MMEB Training Split
VLM2Vec (Jiang et al., 2024b) Phi-3.5-V-4.2B 0.7 54.8 54.9 62.3 79.5 60.1
VLM2Vec (Jiang et al., 2024b) LLaV A-1.6-7B 0.7 61.2 49.9 67.4 86.1 62.9
MMRet (Zhou et al., 2024) LLaV A-1.6-7B 26.9 56.0 57.4 69.9 83.6 64.1
IDMR (Liu et al., 2025a) InternVL2.5-8B 1.2 58.3 58.6 68.7 85.6 64.9
IDMR (Liu et al., 2025a) InternVL2.5-26B 1.2 66.3 61.9 71.1 88.6 69.2
CAFe (Yu et al., 2025) LLaV A-OV-7B 0.9 65.2 65.6 70.0 91.2 69.8
mmE5 (Chen et al., 2025) Llama-3.2-11B-Vision 1.2 67.6 62.6 71.0 89.6 69.8
LLaVE (Lan et al., 2025) Aquila-VL-2B 0.7 62.1 60.2 65.2 84.9 65.2
LLaVE (Lan et al., 2025) LLaV A-OV-7B 0.7 65.7 65.4 70.991.9 70.3
UNITE (Kong et al., 2025) Qwen2-VL-2B 7.9 63.2 55.9 65.4 75.6 63.3
UNITE (Kong et al., 2025) Qwen2-VL-7B 7.9 68.365.171.684.8 70.3
UniME (Gu et al., 2025) Phi-3.5-V-4.2B 0.9 54.8 55.9 64.5 81.8 64.2
UniME (Gu et al., 2025) LLaV A-1.6-7B 0.9 60.6 52.9 67.9 85.1 66.6
UniME (Gu et al., 2025) LLaV A-OV-7B 0.9 66.866.670.6 90.9 70.7
Not Directly Supervised by MMEB
UniME (Gu et al., 2025) Phi-3.5-V-4.2B 0.3 42.5 18.3 40.5 59.9 40.3
UniME (Gu et al., 2025) LLaV A-1.6-7B 0.3 43.0 17.7 42.563.2 41.6
MMRet (Zhou et al., 2024) LLaV A-1.6-7B 26.2 47.2 18.4 56.5 62.2 44.0
MM-Embed (Lin et al., 2024) LLaV A-Next-7B 1.1 48.1 32.3 63.8 57.8 50.0
LamRA-Ret (Liu et al., 2025b) Qwen2.5-VL-7B 1.4 51.7 34.1 66.9 56.7 52.4
GME (Zhang et al., 2024b) Qwen2-VL-2B 8.0 54.4 29.9 66.9 55.5 51.9
GME (Zhang et al., 2024b) Qwen2-VL-7B 8.0 57.7 34.7 71.259.3 56.0
Training-Free Approaches
E5-V† (Jiang et al., 2024a) Qwen2.5-VL-3B – 39.9 30.4 36.6 41.3 36.3
E5-V† (Jiang et al., 2024a) Qwen2.5-VL-7B – 41.2 37.2 37.9 48.4 39.8
FreeRet (ours) LLaV A-OV-7B – 59.6 58.7 62.4 65.9 61.0
FreeRet (ours) LLaV A-OV-1.5-8B – 62.8 69.2 65.9 65.3 65.9
FreeRet (ours) InternVL3-2B – 59.1 58.2 56.2 65.1 58.5
FreeRet (ours) InternVL3-8B – 62.3 68.9 64.1 79.9 66.7
FreeRet (ours) InternVL3-14B – 61.1 71.2 67.9 85.1 68.9
FreeRet (ours) Qwen2-VL-7B – 65.6 64.1 68.7 75.2 67.2
FreeRet (ours) Qwen2.5-VL-3B – 57.5 61.2 62.8 57.8 60.3
FreeRet (ours) Qwen2.5-VL-7B – 69.4 70.0 69.9 78.2 70.7
FreeRet (ours) Qwen2.5-VL-32B – 70.8 75.4 72.4 77.0 73.3
Table 2:Performance comparison on two video-centric tasksfrom the MMEB-V2 bench-
mark (Meng et al., 2025): video classification and video retrieval, each comprising five datasets.
Model MLLM Train data (M) Video data Video Classification Video Retrieval
GME (Zhang et al., 2024b) Qwen2-VL-2B 8.0✗34.9 25.6
GME (Zhang et al., 2024b) Qwen2-VL-7B 8.0✗37.4 28.4
LamRA (Liu et al., 2025b) Qwen2-VL-7B 1.4✗39.3 24.3
VLM2Vec (Jiang et al., 2024b) Qwen2-VL-2B 0.7✗33.4 20.6
VLM2Vec (Jiang et al., 2024b) Qwen2-VL-7B 0.7✗39.1 29.0
VLM2Vec-V2 (Meng et al., 2025) Qwen2-VL-2B 1.7✓39.3 28.8
FreeRet (ours) Qwen2-VL-2B – – 58.3 33.6
FreeRet (ours) Qwen2-VL-7B – – 63.2 39.3
4.2 MAINRESULTS
Tab. 1 summarizes the results on MMEB. Our FreeRet delivers substantial gains over the training-
free baseline E5-V , improving by+34.4%with Qwen2.5-VL-3B and+33.5%with Qwen2.5-VL-
7B. These results highlight the untapped potential of training-free multimodal retrieval and establish
FreeRet as a strong step forward in this paradigm. Methods trained on large-scale data, but without
explicit MMEB supervision, perform notably worse. For instance, MMRet trained on 26.2M sam-
ples reaches only 44.0%, while GME achieves 56.0% despite relying on carefully curated data. This
points to the limited out-of-distribution generalization of post-training approaches. Even when com-
pared to in-domain supervised methods, FreeRet remains highly competitive. With Qwen2.5-VL-
7B, FreeRet approaches the performance of UniME, the current state-of-the-art supervised model.
Scaling further to Qwen2.5-VL-32B, our approach surpasses existing supervised results. On video
tasks (see Tab. 2), FreeRet again achieves strong performance, outperforming all baselines by large
margins on both video classification and video retrieval, without any task-specific training.
Overall, these findings convey two central insights: (1) post-training approaches depend heavily on
in-domain data and deliver limited generalization across distribution shifts; (2) in contrast, FreeRet
demonstrates robust zero-training ability across diverse datasets, tasks, and modalities, setting a new
benchmark for training-free multimodal retrieval.
7

Preprint
Table 3:Ablation study of FreeRetwith Qwen2.5-VL (3B and 7B). Results are reported as Preci-
sion@1, averaged over 36 MMEB datasets. The baseline configuration is highlighted in gray, while
the FreeRet configuration is shown in green.
(a) Alleviate lexicalization pressure.
Embedding 3B 7B
hMLP
L 45.34 47.97
hAttn
L 50.67 53.68
hMLP
L−1 51.0451.03
hMLP
L−2 50.64 48.78(b) Control embedding generation.
Configuration 3B 7B
(a): base instruct Eq. (1) 42.42 45.53
(b): (a) + semantic ground 46.71 50.60
(c): (b) + noise suppress 48.20 51.50
(d): (c) + task align 50.67 53.68(c) Neutralize LLM framing effect.
Label framing 3B 7B
Right-Wrong 59.46 64.71
Yes-No 58.39 65.28
True-False 60.06 66.71
Multiple-choice 60.31 70.72
4.3 ABLATIONSTUDY
Effect of the Final MLP Layer.We first probe the role of the final MLP layer, which we posit
introduces lexicalization pressure that degrades embedding quality. As a baseline, we adopt the
hidden state from the last transformer layer (hMLP
L), a standard choice in prior work. We then ablate
at three depths: (i) bypassing the final MLP by usinghAttn
L, (ii) removing the full last transformer
layer by usinghMLP
L−1, and (iii) removing the last two transformer layers by usinghMLP
L−2. Results
in Tab. 3a show that eliminating only the final MLP yields consistent gains: improving the 3B and
7B models by 5.33% and 5.71%, confirming that lexicalization pressure indeed harms representation
quality. However, discarding additional layers fails to bring further benefits; the 7B variant even
degrades. We hypothesize that the deeper layers capture essential semantic abstractions. The effect
is magnified in shallower architectures such as Qwen2.5-VL 7B (28 layers) compared to the 3B
variant (36 layers).
Controlling Embeddings via Prompts.We then investigate whether embeddings can be guided
at inference without altering parameters. Starting from a simple instruction (Eq. (1)), we progres-
sively add lightweight constraints (see Tab. 3b). Explicit semantic grounding substantially improves
alignment, yielding 4.29% and 5.07% gains for 3B and 7B models. Adding noise-suppression in-
structions further attenuates spurious function words, giving another increase of 1.49% and 0.9%.
Finally, encoding task-specific priors yields an additional boost of 2.47% and 2.17%. Together, these
results demonstrate that the embedding space of large models can be effectively reshaped through
prompting control, offering a parameter-free avenue for fine-grained representation steering.
Study on the LLM Framing Effect.Next, we test whether the surface framing of reranking
questions impacts performance. As shown in Tab. 3c, semantically neutral labels such asTrue/False
consistently outperform alternatives with that carry social or pragmatic connotations, such asYes/No
orRight/Wrong. Our multiple-choice framing achieves the highest accuracy, due to its neutrality as
well as consistency with pretraining distributions, where such formats are frequent. Interestingly,
sensitivity increases with model scale. The 7B variant shows large variance across framings, while
the 3B model remains relatively stable. We conjecture that larger models capture finer semantic
distinctions and are thus more vulnerable to subtle framing shifts, whereas smaller models operate
on coarser abstractions less affected by such perturbations.
w/o
rerankingw/o
reranking
Figure 5: Varying the number of reranking candidates.Impact of the Reranking Stage.
Finally, we examine the contribution
of our reranking stage. Prior mul-
timodal retrievers typically optimize
the embedding model while underex-
ploring reranking, yet accurate can-
didate selection heavily depends on
this step. In Fig. 5, we vary the
number of candidates passed to the
reranking stage. Benchmark perfor-
mance consistently improves as the reranking pool enlarges, underlining its necessity. Notably, our
FreeRet reuses the same model across both embedding and reranking, avoiding additional parame-
terization or deployment overhead. This yields a favorable balance between precision and efficiency,
positioning reranking as an indispensable yet economical design choice.
8

Preprint
0.8240.748My mom always  said, life was like a 
box of chocolates . You never  know  
what  you're  gonna  get. Those  must  
be comfortable  shoes . I'll bet you 
could  walk all day in shoes  like that 
and not feel a thing . I wish I had 
shoes  like that. My feet hurt. Mom 
always  said, there's  an awful  lot you 
could  tell about  a person  by their  
shoes . Where  they go. Where  
they've  been . I've worn  lots of 
shoes . I bet if I think  about  it real 
hard,  I could  remember  my first  
pair of shoes . Mama  said they'd  take 
me anywhere .
0.7580.727
0.5310.349
aaaQuery (audio) 
 aaaaSearch  pool (video)
Candidate  
Searching
Precise 
Reranking
The person  in 
the picture  are 
participating  in 
a TV show .
FreeRet -EmbedderSelect Top -2
FreeRet -Reranker
Response
to query
0.6600.668
0.8090.711
0.9530.797
aaaQuery (image+text )
Candidate  
Searching
Precise 
Reranking
FreeRet -EmbedderSelect Top -2
FreeRet -Reranker
Response
to query
aaaaSearch  pool (video)
Figure 6:FreeRet enables instant omni-modal retrievalwith omni-modality models. Illustrated
with Qwen2.5-Omni: audio-to-video retrieval (left); image+text to video retrieval (right).
4.4 DISCUSSIONS ONTRAINING-FREEADVANTAGES
Instant Deployment.A key strength of the training-free paradigm is its ability to turn any MLLM
into a retriever immediately, with no additional fine-tuning. This property allows practitioners to
flexibly leverage diverse model families and scales depending on task requirements. Given the
rapid pace of new MLLM releases, the cost of repeated fine-tuning quickly becomes prohibitive.
By eliminating this overhead, our approach enables faster adoption of new advances. Moreover,
the training-free nature broadens MLLM evaluation beyond conventional QA objectives to include
retrieval-oriented benchmarks, offering a richer assessment of their multimodal reasoning capacity.
Scalable Omni-Modality Retrieval.Training-based multimodal retrieval encounters a funda-
mental scalability bottleneck: each new modality demands extensive paired data covering all
query–target configurations. The data requirement grows combinatorially. For example, with four
modalities (text, image, video, audio), there existP4
i=1 4
i
= 15possible modality combinations,
leading to15×15 = 225query–target pair types. This cost escalates further as modalities increase,
rendering full coverage infeasible in practice. Our method sidesteps this limitation by directly ex-
ploiting the intrinsic omni-modal understanding already embedded in MLLMs. As shown in Fig. 6,
FreeRet supports retrieval across arbitrary configurations,e.g., retrieving a video using an audio
query or a joint image–text query, even when the model has never been trained on such pairs. In this
way, omni-modality retrieval shifts from a data-intensive challenge to a natural emergent capability.
Preserving Multimodal Intelligence.Unlike fine-tuning pipelines that risk eroding a model’s
pretrained strengths, FreeRet preserves the multimodal reasoning and conversational abilities of the
underlying MLLM. Retrieval is introduced as an additional functionality, not as a replacement. Con-
sequently, the pretrained capabilities remain intact while retrieval emerges as a first-class operation.
This design enables a single model to natively support retrieval, reranking, and generation within a
unified RAG framework, avoiding the fragmentation introduced by specialized expert modules. The
result is both a reduction in engineering complexity and an increase in deployment efficiency.
Table 4:Performance gains on
LVBench: Qwen2.5-VL with FreeRet.
Sample Method # Frames LVBench
Uniform 64 39.0
FreeRet 64 44.8 (+5.8)
Uniform 32 39.0
FreeRet 32 44.2 (+5.2)
Uniform 16 36.7
FreeRet 16 42.7 (+6.0)Towards Long-Video Understanding.Reasoning over
long videos is particularly challenging due to extended
temporal contexts, where uniform frame sampling often
dilutes attention with redundant content. FreeRet of-
fers a pragmatic solution: it first retrieves the most rele-
vant frames, thereby grounding subsequent reasoning on
evidence-rich content. This retrieval-driven focus effec-
tively reduces temporal redundancy. On the hour-level
benchmark LVBench (Wang et al., 2024b), experiments with Qwen2.5-VL 7B (see Tab. 4) demon-
strate consistent improvements. These results highlight the promise of FreeRet as a foundation for
scaling MLLMs toward long-horizon multimodal reasoning.
9

Preprint
5 CONCLUSION
This work demonstrates that pretrained MLLMs can act as effective retrieval engines without any
additional training. By decoupling embedding extraction from lexical alignment, conditioning rep-
resentation with explicit priors, and neutralizing framing in reranking, FreeRet turns off-the-shelf
MLLMs into strong two-stage retrievers. Experiments on MMEB show that FreeRet not only sur-
passes heavily trained baselines but also remains competitive with MMEB-supervised methods. Be-
yond performance, its plug-and-play nature preserves reasoning ability, supports arbitrary modality
combinations, and integrates retrieval with generation in a single model. These results challenge the
prevailing reliance on costly contrastive training and point toward a retrieval paradigm where gen-
eralist MLLMs serve as unified, training-free backbones for multimodal reasoning and generation.
6 REPRODUCIBILITYSTATEMENT
FreeRet is implemented within a widely adopted search-then-reranking retrieval framework. In the
search stage, we show our prompting strategies in § 3.3 and Fig. 3, and extract features following
theRemedyprocedure described in § 3.2. For the reranking stage, our process is detailed in § 3.4,
covering both multiple-choice question framing and score computation. To further support repro-
ducibility, we provide runnable code in the supplemental material.
REFERENCES
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report.arXiv preprint arXiv:2303.08774, 2023.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language
model for few-shot learning.Advances in neural information processing systems, 35:23716–
23736, 2022.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report.arXiv preprint arXiv:2502.13923,
2025.
Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hul-
lender. Learning to rank using gradient descent. InProceedings of the 22nd international confer-
ence on Machine learning, pp. 89–96, 2005.
Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou.
mme5: Improving multimodal multilingual embeddings via high-quality synthetic data.arXiv
preprint arXiv:2502.08468, 2025.
Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shen-
glong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source
multimodal models with model, data, and test-time scaling.arXiv preprint arXiv:2412.05271,
2024.
LLaV A Community Contributors. Llava-onevision-1.5: Fully open framework for democratized
multimodal training. Inarxiv, 2025.
Yuhao Cui, Xinxing Zu, Wenhua Zhang, Zhongzhou Zhao, and Jinyang Gao. Incorporating dense
knowledge alignment into unified multimodal representation models. InProceedings of the Com-
puter Vision and Pattern Recognition Conference, pp. 29733–29743, 2025.
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hi-
erarchical image database. In2009 IEEE conference on computer vision and pattern recognition,
pp. 248–255. Ieee, 2009.
10

Preprint
Yuchen Fu, Zifeng Cheng, Zhiwei Jiang, Zhonghui Wang, Yafeng Yin, Zhengliang Li, and Qing Gu.
Token prepending: A training-free approach for eliciting better sentence embeddings from llms.
arXiv preprint arXiv:2412.11556, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1), 2023.
Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long,
Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal em-
bedding learning with multimodal llms.arXiv preprint arXiv:2504.17432, 2025.
Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padman-
abhan, Giuseppe Ottaviano, and Linjun Yang. Embedding-based retrieval in facebook search.
InProceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, pp. 2553–2561, 2020.
Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. Scaling sentence
embeddings with large language models.arXiv preprint arXiv:2307.16645, 2023.
Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang,
Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language
models.arXiv preprint arXiv:2407.12580, 2024a.
Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec:
Training vision-language models for massive multimodal embedding tasks.URL https://arxiv.
org/abs/2410.05160, 2024b.
Yeong-Joon Ju and Seong-Whan Lee. From generator to embedder: Harnessing innate abilities
of multimodal llms via building zero-shot discriminative embedding model.arXiv preprint
arXiv:2508.00955, 2025.
Fanheng Kong, Jingyuan Zhang, Yahui Liu, Hongzhi Zhang, Shi Feng, Xiaocui Yang, Daling Wang,
Yu Tian, Fuzheng Zhang, Guorui Zhou, et al. Modality curation: Building universal embeddings
for advanced multimodal information retrieval.arXiv preprint arXiv:2505.19650, 2025.
Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Llave: Large language
and vision embedding models with hardness-weighted contrastive learning.arXiv preprint
arXiv:2503.04812, 2025.
Yibin Lei, Di Wu, Tianyi Zhou, Tao Shen, Yu Cao, Chongyang Tao, and Andrew Yates. Meta-task
prompting elicits embeddings from large language models.arXiv preprint arXiv:2402.18458,
2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan
Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer.arXiv preprint
arXiv:2408.03326, 2024.
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-
training for unified vision-language understanding and generation. InInternational conference on
machine learning, pp. 12888–12900. PMLR, 2022.
Ziyue Li and Tianyi Zhou. Your mixture-of-experts llm is secretly an embedding model for free.
arXiv preprint arXiv:2410.10814, 2024.
Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei
Ping. Mm-embed: Universal multimodal retrieval with multimodal llms.arXiv preprint
arXiv:2411.02571, 2024.
11

Preprint
Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie,
and Chaochao Lu. Idmr: Towards instance-driven precise visual correspondence in multimodal
retrieval.arXiv preprint arXiv:2504.00954, 2025a.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning.Advances
in neural information processing systems, 36:34892–34916, 2023.
Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and
Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. InProceedings
of the Computer Vision and Pattern Recognition Conference, pp. 4015–4025, 2025b.
Yibo Lyu, Rui Shao, Gongwei Chen, Yijie Zhu, Weili Guan, and Liqiang Nie. Puma: Layer-pruned
language model for efficient unified multimodal retrieval with modality-adaptive learning.arXiv
preprint arXiv:2507.08064, 2025.
Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran
Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images,
and visual documents.arXiv preprint arXiv:2507.04590, 2025.
Bhaskar Mitra, Fernando Diaz, and Nick Craswell. Learning to match using local and distributed
representations of text for web search. InProceedings of the 26th international conference on
world wide web, pp. 1291–1299, 2017.
Yassine Ouali, Adrian Bulat, Alexandros Xenos, Anestis Zaganidis, Ioannis Maniadis Metaxas,
Brais Martinez, and Georgios Tzimiropoulos. Vladva: Discriminative fine-tuning of lvlms. In
Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 4101–4111, 2025.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. InInternational conference on machine learning, pp.
8748–8763. PmLR, 2021.
Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz
Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. Recommender systems with gener-
ative retrieval.Advances in Neural Information Processing Systems, 36:10299–10315, 2023.
Benjamin Schneider, Florian Kerschbaum, and Wenhu Chen. Abc: Achieving better control of
multimodal embeddings using vlms.arXiv preprint arXiv:2503.00329, 2025.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented
generation: A survey on agentic rag.arXiv preprint arXiv:2501.09136, 2025.
Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Rep-
etition improves language model embeddings.arXiv preprint arXiv:2402.15449, 2024.
Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training
techniques for clip at scale.arXiv preprint arXiv:2303.15389, 2023.
Raghuveer Thirukovalluru and Bhuwan Dhingra. Geneol: Harnessing the generative power of llms
for training-free sentence embeddings.arXiv preprint arXiv:2410.14635, 2024.
Nam V o, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text
and image for image retrieval-an empirical odyssey. InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 6439–6448, 2019.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu,
Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the
world at any resolution.arXiv preprint arXiv:2409.12191, 2024a.
Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin
Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding
benchmark. InCVPR2025, 2024b.
12

Preprint
Zhen Wang, Xu Shan, Xiangxie Zhang, and Jie Yang. N24news: A new dataset for multimodal
news classification.arXiv preprint arXiv:2108.13327, 2021.
Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, M ´arton Kardos, Roman
Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, and Niklas Muennighoff. Mieb: Massive
image embedding benchmark.arXiv preprint arXiv:2504.10471, 2025.
Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang
Fan, Kai Dang, et al. Qwen2. 5-omni technical report.arXiv preprint arXiv:2503.20215, 2025a.
Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, and
Hengxing Cai. Mm-r5: Multimodal reasoning-enhanced reranker via reinforcement learning for
document retrieval.arXiv preprint arXiv:2506.12364, 2025b.
Youze Xue, Dian Li, and Gang Liu. Improve multi-modal embedding learning via explicit hard
negative gradient amplifying.arXiv preprint arXiv:2506.02020, 2025.
Hao Yu, Zhuokai Zhao, Shen Yan, Lukasz Korycki, Jianyu Wang, Baosheng He, Jiayi Liu, Lizhu
Zhang, Xiangjun Fan, and Hanchao Yu. Cafe: Unifying representation and generation with
contrastive-autoregressive finetuning.arXiv preprint arXiv:2503.19900, 2025.
Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
European conference on computer vision, pp. 818–833. Springer, 2014.
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language
image pre-training. InProceedings of the IEEE/CVF international conference on computer vision,
pp. 11975–11986, 2023.
Bowen Zhang, Kehua Chang, and Chunping Li. Simple techniques for enhancing sentence embed-
dings in generative language models. InInternational Conference on Intelligent Computing, pp.
52–64. Springer, 2024a.
Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan
Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal
llms.arXiv preprint arXiv:2412.16855, 2024b.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint arXiv:2506.05176, 2025.
Pengfei Zhao, Rongbo Luan, Wei Zhang, Peng Wu, and Sifeng He. Guiding cross-modal represen-
tations with mllm priors via preference alignment.arXiv preprint arXiv:2506.06970, 2025.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving
few-shot performance of language models. InInternational conference on machine learning, pp.
12697–12706. PMLR, 2021.
Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian,
and Yongping Xiong. Megapairs: Massive data synthesis for universal multimodal retrieval.arXiv
preprint arXiv:2412.14475, 2024.
Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen
Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for
open-source multimodal models.arXiv preprint arXiv:2504.10479, 2025.
13

Preprint
A APPENDIX
Figure 7: Qwen2.5-VL 3B and 32B results in probing experiments on lexicalization pressure.
Table 5: Ablate the choice of removing the final MLP layer on models beyond Qwen. We report
average Precision@1 scores on the MMEB benchmark.
Embedding LLaV A-OV-1.5-8B InternVL3-2B InternVL3-8B
hMLP
L 47.8 40.2 44.2
hAttn
L 54.2 (+6.4) 46.0 (+5.8) 49.7 (+5.5)
A.1 ADDITIONALDETAILSONLEXICALIZATIONPRESSUREEXPERIMENTS
We investigate the locus and consequences of lexicalization in multi-modal large language mod-
els through probing experiments across different model scales. In the main text, we report results
on Qwen2.5-VL 7B, showing that lexicalization is strongly concentrated in the final MLP layers.
This concentration appears to sharpen token-level alignment, but it comes at the cost of semantic
coherence, suggesting that lexicalization exerts a structural pressure on representation quality.
To validate the robustness of this finding, we further evaluate Qwen2.5-VL 3B and 32B. As shown
in Fig. 7, all models exhibit consistent patterns: the final MLP absorbs the majority of lexicalization
load and induces similar trade-offs between lexical grounding and semantic fidelity. This conver-
gence across scales indicates that lexicalization pressure is not an artifact of model size, but instead
emerges as a general structural property of the architecture.
Our evaluation involves 720 data points, sampled systematically from MMEB. Specifically, we se-
lect 20 examples from each of the 36 datasets. For all reported scores, we compute both the mean
and the standard deviation, ensuring the conclusions are driven by stable trends rather than dataset-
specific artifacts.
Moreover, we extend our analysis beyond the Qwen series by evaluating additional architectures,
including InternVL and LLaV A. The results, summarized in Tab. 5, demonstrate that lexicalization
pressure is a pervasive phenomenon. Importantly, our simple modification (removing the final MLP
layer) consistently improves performance across diverse MLLM architectures.
A.2 ADDITIONALTHEORETICALANALYSIS
We provide a simplified argument illustrating why discarding the final MLP layer can preserve
semantic fidelity of hidden states.
14

Preprint
Lemma 1(Lexicalization Alignment).Leth∈Rdbe the hidden state before the last MLP , and
define
h′=Ah+b, A∈Rd×d, b∈Rd.
LetW= [w 1, . . . ,w |V|]∈Rd×|V|be the LM head and
L(h′) =−logexp(w⊤
y∗h′)P
v∈Vexp(w⊤vh′)
the cross-entropy loss. Then the gradient is
∇h′L=X
v∈Vp(v|h′)wv−wy∗,
which points toward aligningh′withw y∗while suppressing components orthogonal tospan(W).
Proof.Since∇ h′Lis always a linear combination of vocabulary vectors{w v}, optimization over
Ldepends only on the projection ofh′ontoW= span{w v}. Any orthogonal componentϵ=
h′−PW(h′)satisfies∇ h′L⊤ϵ= 0and is thus uncontrolled by the objective. Training therefore
drivesh′to maximize⟨h′,wy∗⟩while diminishing the effective role ofϵ.
Corollary A.0.1(Lexicalization Pressure).The final MLP layer learnsh′≈PW(h), which in-
creases alignment with the lexical head but reduces retention of semantic features outsideW.
Remark(Practical Implication).By discarding the final MLP when producing embeddings, we by-
pass the forced projection intoW, thereby retaining semantic components ofhthat would otherwise
be suppressed by lexicalization pressure. This aligns with our empirical findings in Fig. 2b.
A.3 ADDITIONALDETAILSONWORD-LEVELPROBABILITYVISUALIZATION
To better understand the representational behavior of our model, we visualize word-level probabil-
ities as shown in Fig. 3. Although our main method removes the final MLP layer to ensure more
faithful embeddings, we reintroduce this layer solely for visualization purposes. Specifically, we
pass the hidden states through the original language modeling head followed by a softmax, which
produces interpretable token-wise probabilities.
This procedure allows us to probe the model’s internal distribution over the vocabulary without
altering the underlying training or inference pipeline. Compared with E5V , the key difference lies in
our prompt design, which directly shapes the representation generation process. This design choice
provides a clearer window into how our approach influences semantic alignment, highlighting the
improvements our method achieves over prior baselines.
A.4 ADDITIONALDETAILS ONLLM FRAMINGEFFECTEXPERIMENTS
For benchmark evaluation, we employ the ImageNet-1K (Deng et al., 2009) subset from the MMEB
benchmark. To ensure robustness, we rephrase the shared prefix (e.g., the prompt question) three
times and report the average accuracy across these variants.
For the context-free instruction setting, we further mitigate position-related biases by swapping the
order of the labels. For instance, we alternate between instructions such as “Reply only with
‘Yes’ or ‘No’” and “Reply only with ‘No’ or ‘Yes’”. We then average the corre-
sponding logits to minimize the influence of positional effects in the instruction text.
A.5 THEPOTENTIAL OFFREERET INMASSIVE-CANDIDATESCENARIOS
In large-scale retrieval, inference efficiency is often as critical as accuracy. While FreeRet, built
upon modern MLLMs, achieves strong performance, it incurs higher latency compared to CLIP-
based methods, a limitation also noted in prior MLLM-based retrievers (Lin et al., 2024; Liu et al.,
2025b; Cui et al., 2025).
To ensure scalability in scenarios with over 100M candidates (a setting relevant to many real-
world applications), we propose three orthogonal strategies, readily applicable to FreeRet and other
MLLM-based retrievers:
15

Preprint
1.Coarse pre-filtering.Employing an extremely lightweight model for initial filtering re-
duces the effective candidate pool before invoking MLLM-based retrieval and reranking.
2.Controlled reranking.Since reranking dominates inference cost, limiting the number
of reranked candidates yields significant efficiency gains with minimal performance loss
(see Fig. 5).
3.Lightweight MLLMs.Substituting the backbone with more efficient MLLMs provides
further savings, and ongoing progress in model compression suggests increasingly favor-
able trade-offs in the near future.
A.6 LIMITATIONS
As discussed in § A.5, FreeRet inherits the computational overhead of MLLM-based retrievers,
which may limit its practicality in resource-constrained environments. Furthermore, unlike data-
driven methods that can adapt through task-specific training, FreeRet is entirely training-free and
relies solely on the underlying multimodal understanding and instruction-following capability of the
foundation model. Consequently, its performance may degrade when the base model itself provides
weak representations or multimodal reasoning ability.
16