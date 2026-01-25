# LLMOrbit: A Circular Taxonomy of Large Language Models -From Scaling Walls to Agentic AI Systems

**Authors**: Badri N. Patro, Vijay S. Agneeswaran

**Published**: 2026-01-20 15:06:19

**PDF URL**: [https://arxiv.org/pdf/2601.14053v1](https://arxiv.org/pdf/2601.14053v1)

## Abstract
The field of artificial intelligence has undergone a revolution from foundational Transformer architectures to reasoning-capable systems approaching human-level performance. We present LLMOrbit, a comprehensive circular taxonomy navigating the landscape of large language models spanning 2019-2025. This survey examines over 50 models across 15 organizations through eight interconnected orbital dimensions, documenting architectural innovations, training methodologies, and efficiency patterns defining modern LLMs, generative AI, and agentic systems. We identify three critical crises: (1) data scarcity (9-27T tokens depleted by 2026-2028), (2) exponential cost growth ($3M to $300M+ in 5 years), and (3) unsustainable energy consumption (22x increase), establishing the scaling wall limiting brute-force approaches. Our analysis reveals six paradigms breaking this wall: (1) test-time compute (o1, DeepSeek-R1 achieve GPT-4 performance with 10x inference compute), (2) quantization (4-8x compression), (3) distributed edge computing (10x cost reduction), (4) model merging, (5) efficient training (ORPO reduces memory 50%), and (6) small specialized models (Phi-4 14B matches larger models). Three paradigm shifts emerge: (1) post-training gains (RLHF, GRPO, pure RL contribute substantially, DeepSeek-R1 achieving 79.8% MATH), (2) efficiency revolution (MoE routing 18x efficiency, Multi-head Latent Attention 8x KV cache compression enables GPT-4-level performance at <$0.30/M tokens), and (3) democratization (open-source Llama 3 88.6% MMLU surpasses GPT-4 86.4%). We provide insights into techniques (RLHF, PPO, DPO, GRPO, ORPO), trace evolution from passive generation to tool-using agents (ReAct, RAG, multi-agent systems), and analyze post-training innovations.

## Full Text


<!-- PDF content starts -->

LLMOrbit: A Circular Taxonomy of Large Language Models
—From Scaling Walls to Agentic AI Systems
Badri N. Patro, Vijay S. Agneeswaran
Microsoft
{badripatro, vagneeswaran}@microsoft.com
Abstract
The field of artificial intelligence has undergone a rev-
olution from foundational Transformer architectures to
reasoning-capable systems approaching human-level per-
formance on certain specific tasks. We presentLLMOr-
bit, a comprehensive circular taxonomy navigating the
complete landscape of large language models spanning
2019-2025. This survey examines over 50 major mod-
els across 15 organizations through eight interconnected
orbital dimensions, documenting the architectural inno-
vations, training methodologies, and efficiency patterns
defining modern large language models (LLMs), genera-
tive AI, and agentic systems.
The Scaling Wall:We identify three critical crises
threatening AI progress—(1) data scarcity (human gen-
erated text data is approximately 9-27T tokens, which
could be depleted by 2026-2028), (2) exponential cost
growth ($3M to $300M+ cost to train LLM models in 5
years), and (3) unsustainable energy consumption (22×
increase)—establishing the ”scaling wall” that fundamen-
tally limits brute-force scaling.
Breaking the Scaling Wall:Our analysis reveals
six different paradigms: (1)test-time compute—o1 and
DeepSeek-R1 achieve GPT-4 performance with 10×in-
ference compute but smaller pre-training; (2)quanti-
zation—4-8×compression with ¡1% perplexity degra-
dation [167, 43]; (3)distributed edge computing—10×
cost reduction [164]; (4)model merging—synergistic ca-
pabilities [53, 147]; (5)efficient training—ORPO re-
duces memory 50% [65]; (6)small specialized mod-
els—Microsoft Phi-4 (14B) matches 10×larger mod-
els [89].Three fundamental paradigm shifts emerge: (1)post-
training gains—recent work suggests post-training tech-
niques (RLHF, GRPO, pure RL) contribute substan-
tially to model capabilities [109, 12], with DeepSeek-R1
achieving 79.8% MATH through pure RL without super-
vised fine-tuning; (2)efficiency revolution—MoE rout-
ing (18×efficiency), Multi-head Latent Attention (8×
KV cache compression) enable GPT-4-level performance
at ¡$0.30/M tokens (100×cost reduction); (3)democ-
ratization—open-source Llama 3 (88.6% MMLU) sur-
passes GPT-4 (86.4%), while global models from Al-
ibaba (Qwen), Moonshot (Kimi), Microsoft (Phi), and
Zhipu (GLM) demonstrate competitive capabilities. We
provide insight into key techniques (RLHF, PPO, DPO,
GRPO, ORPO), trace evolution from passive generation
to active tool-using agents (ReAct, RAG, multi-agent sys-
tems), and analyze post-training innovations (quantiza-
tion, merging, edge deployment). Through benchmark-
ing across 9 metrics, we identify reasoning emergence
requires: (i) scale exceeding1011tokens, (ii) RL with
verifiable feedback, and (iii) test-time search. LLMOrbit
serves as both a technical reference for practitioners and a
roadmap for researchers exploring reasoning, multimodal
understanding, and autonomous agents in the post-scaling
era.
1 Introduction
The field of artificial intelligence has undergone a re-
markable transformation over the past decade, driven pri-
marily by breakthrough advances in large language mod-
els (LLMs), generative AI, and agentic systems. From
1arXiv:2601.14053v1  [cs.LG]  20 Jan 2026

ReAct
Planning
Tool Use
Memory
CoT/ToTSingle Agent
Multi-Agent
RAGTraining Method
RLHF PPO
DPO GRPO
ORPO Reward Models
PEFT EnvironmentModel Evolution Scaling Wall Breaking Wall
Architecture Benchmarking EnvironmentalAgentic AI
GenAI
LLMLLMOrbit: From Foundation to Agentic AIFigure 1:Evolution from LLM Foundation to Agentic AI.Three nested paradigms converge to a unified frame-
work:LLM Foundation(blue) encompasses model evolution, scaling challenges, and architectural innovations;GenAI
(purple) adds training methodologies (RLHF, PPO, DPO, GRPO, ORPO) and environments;Agentic AI(light blue)
extends capabilities through reasoning (ReAct, CoT/ToT), tool use (RAG), and multi-agent systems. This nested
architecture illustrates how foundation models serve as the base for generative capabilities, which in turn enable au-
tonomous agentic systems.
the foundational Transformer architecture [150] to the re-
cent emergence of Large Reasoning Models (LRMs) like
GPT-4o, Claude 3.5, DeepSeek-R1, and Gemini 1.5, the
landscape has evolved at an unprecedented pace (Fig-
ure 1). This survey, which we callLLMOrbit, provides
a comprehensive circular taxonomy navigating the com-
plete LLM landscape through eight interconnected orbital
dimensions (Figure 2): scaling challenges and paradigms
that break the scaling wall, model taxonomy which covers
50+ models, training methodologies from RLHF to pure
reinforcement learning, architectural innovations from
Transformers to state space models, efficiency techniques
and compression, agentic AI frameworks, rigorous bench-
marking protocols, and economic-environmental consid-
erations. LLMOrbit synthesizes foundational models and
agentic systems, providing both technical depth and prac-
tical deployment insights for researchers and practitioners
navigating this rapidly evolving landscape.
1.1 The Foundation: Transformers and Ar-
chitectural Breakthroughs
The introduction of the Transformer architecture in
2017 [150] marked a paradigm shift in natural lan-
guage processing. By replacing recurrent mechanisms
with self-attention, the Transformer enabled parallel pro-cessing of sequences and better capture of long-range
dependencies through multi-head attention mechanisms.
The core innovation lies in the scaled dot-product at-
tention: Attention(Q, K, V) =softmax(QKT
√dk)V, where
Q,K, andVrepresent query, key, and value matrices.
This architectural breakthrough laid the groundwork for
all subsequent large language models, including GPT,
BERT [44], T5 [120], and modern multimodal systems
like CLIP [116].
The attention mechanism itself was first introduced by
Bahdanau et al. [11] (2015) for neural machine transla-
tion, enabling models to dynamically focus on relevant
input segments. The Transformer generalized this to self-
attention, where each token attends to all other tokens
in the sequence, creating rich contextual representations.
Subsequent optimizations like FlashAttention [32] and
FlashAttention-2 [31] reduced the quadratic complexity
bottleneck, enabling context lengths exceeding 128K to-
kens.
Alternative architectures have also emerged as potential
successors. Mamba [57] introduced selective state space
models (SSMs) that achieve linear complexity with re-
spect to sequence length, while maintaining competitive
performance on tasks that involve long-range dependen-
cies. The architecture uses selective mechanisms to deter-
mine which inputs to focus on:h t=¯Aht−1+¯Bxt, where
2

Scaling Wall Model Taxonomy Training Architecture
Alternatives Agentic AI Benchmarking EconomicLLMOrbit Taxonomy
Foundation of Agentic AI
(2019 2025)
1. Scaling Wall Analysis2. Model Taxonomy
3. Training Methodology
4. Architecture Evolution
5. Alternative Scaling6. Agentic AI7. Benchmarking8. Economic & EnvironmentalData scarcity Training costs
Energy & carbon Cost modelsGPT-4 Gemini 1.5 Claude 3.5
Llama 3 DeepSeek-V3 Qwen 3
Mistral Phi-4 Nemotron
RLHF PPO DPO
GRPO ORPO Pure RL (R1)
FlashAttention GQA MoE
Mamba RetNet Linear Attn
CLIP Gemini GPT-4o / SoraTest-time compute MoE sparsity
Quantization laws Edge compute
Model merging Efficient training
Small models CompressionCoT ToT ReAct
RAG Tool use Multi-agentMMLU MATH GPQA
HumanEval GSM8K AIME
MT-Bench AlpacaEval LiveCodeBenchHardware cost Amortization
Energy (TDP×PUE) Cloud vs. ownFigure 2:LLMOrbit: A Circular Taxonomy of Large Language Models (2019-2025).This circular orbital archi-
tecture presents eight interconnected dimensions navigating the complete LLM landscape:(1) Scaling Wall Analysis
examining data scarcity, training costs, and energy consumption with quantitative projections;(2) Model Taxonomy
covering 50+ foundation models including GPT-4, Gemini 1.5, Claude 3.5, Llama 3, DeepSeek-V3, Qwen 3, Mis-
tral, Phi-4, and Nemotron;(3) Training Methodologyencompassing RLHF, PPO, DPO, GRPO, ORPO, and pure
reinforcement learning (DeepSeek-R1);(4) Architecture Evolutionfeaturing FlashAttention, Grouped Query At-
tention (GQA), Mixture-of-Experts (MoE), Mamba state space models, RetNet, Linear Attention, and multimodal
systems (CLIP, Gemini, GPT-4o, Sora);(5) Paradigms that Break the Scaling Wallincluding test-time compute
scaling, MoE sparsity (18×efficiency), quantization scaling laws, distributed edge computing, model merging, ef-
ficient training algorithms, small specialized models, and post-training compression;(6) Agentic AI Frameworks
with Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), ReAct, Retrieval-Augmented Generation (RAG), tool use,
and multi-agent systems;(7) Benchmarking Analysisacross MMLU, MATH, GPQA, HumanEval, GSM8K, AIME,
MT-Bench, AlpacaEval, and LiveCodeBench; and(8) Economic & Environmentalconsiderations including hard-
ware costs, amortization formulas, energy metrics (TDP×PUE), and cloud versus on-premise deployment trade-offs.
LLMOrbit synthesizes the complete landscape from foundational models to agentic AI systems, highlighting technical
innovations, evaluation protocols, and practical deployment challenges.
3

¯Aand ¯Bare input-dependent. RetNet [145] proposed re-
tention mechanisms that combine the training parallelism
of Transformers with the efficient inference of RNNs.
1.2 Technical Contributions and Critical
Analysis
This comprehensive LLMOrbit taxonomy is organized
around the eight key orbital dimensions illustrated in Fig-
ure 2, providing both breadth across the LLM landscape
and depth in critical technical dimensions. Understanding
these foundational models, training methodologies, and
architectural innovations builds crucial judgment about
why AI systems work, their limitations, and future direc-
tions. Our analysis makes the following contributions:
1.Scaling Wall Analysis: We provide the first com-
prehensive survey examining the three critical bot-
tlenecks facing continued AI scaling: (1) data
scarcity with projected exhaustion of high-quality
text by 2026-2028 [151, 49], (2) exponentially ris-
ing training costs from $3.3M (GPT-3) to $110M+
(DeepSeek-V3) representing 100×cost explosion in
5 years [15], and (3) energy consumption growing
from 280 MWh (GPT-3) to 6,150 MWh (GPT-4),
with detailed cost breakdowns including hardware
amortization, energy calculations, and cloud com-
pute models.
2.Comprehensive Model Taxonomy: We provide a
detailed technical analysis of over 50 major mod-
els from 2019-2025, documenting their core innova-
tions, architectural choices, training methodologies,
and computational requirements. This includes fron-
tier models (GPT-4, Gemini 1.5, Claude 3.5), open-
source leaders (Llama 3, DeepSeek-V3, Qwen 3),
efficiency-optimized variants (Mistral, Phi-4), and
specialized models (Nemotron), with emphasis on
2025 innovations and emerging trends.
3.Training Methodology Deep Dive: We present
rigorous mathematical formulations and compara-
tive analysis of modern training techniques: Rein-
forcement Learning from Human Feedback (RLHF),
Proximal Policy Optimization (PPO), Direct Pref-
erence Optimization (DPO), Group Relative PolicyOptimization (GRPO), Odds Ratio Preference Opti-
mization (ORPO), and pure reinforcement learning
approaches. The analysis of DeepSeek-R1’s pure
RL methodology provides new insights into reason-
ing emergence without supervised fine-tuning, while
ORPO demonstrates 50% memory reduction through
efficient training [65].
4.Architectural Evolution and Efficiency Innova-
tions: We trace the complete evolution from basic
Transformers to modern efficient variants: FlashAt-
tention and FlashAttention-2 reducing memory bot-
tlenecks, Grouped Query Attention (GQA) enabling
longer contexts, Mixture-of-Experts (MoE) achiev-
ing sparse computation, alternative architectures
(Mamba state space models, RetNet, Linear Atten-
tion), and multimodal extensions (CLIP, Gemini,
GPT-4o, Sora). Special emphasis on 2025 innova-
tions including Multi-Latent Attention (DeepSeek-
V3), Gated DeltaNet (Kimi Linear), and aggressive
sliding window attention strategies.
5.Alternative Scaling Paradigms—Breaking
Through the Wall: We analyze eight emerging
approaches to overcome scaling limitations: (1)test-
time compute scalingwhere o1 and DeepSeek-R1
trade inference compute for reasoning capability,
(2)MoE-based sparsityachieving 18×efficiency
gains (DeepSeek-V3: 671B parameters, 37B ac-
tive), (3)post-training quantizationwith scaling
laws enabling 4-8×compression [167], (4)dis-
tributed edge computingleveraging massive device
networks [164], (5)model mergingcombining
specialized capabilities [53, 147], (6)efficient
training algorithmsreducing memory requirements,
(7)small specialized modelsproving data qual-
ity over scale (Phi-4: 14B parameters matching
larger models) [89], and (8)post-training compres-
sion(QLoRA, GPTQ, AWQ) enabling consumer
hardware deployment.
6.Economic and Environmental Analysis: We pro-
vide detailed cost models for training and deploy-
ment, including hardware acquisition costs ($10K-
30K per GPU), amortization formulas accounting
for depreciation and utilization, energy consump-
tion calculations (TDP×utilization×PUE), and
4

comprehensive cloud versus ownership trade-offs.
This enables practitioners to make informed deci-
sions about model training strategies, deployment ar-
chitectures, and resource allocation for both research
and production scenarios.
7.Comprehensive Benchmarking Analysis: We
compare 30+ models across 9 major benchmarks
with 2025 results, identifying trends in reasoning
capabilities, open-source progress, and efficiency
improvements. Special focus on reasoning bench-
marks (MATH, GPQA, AIME) where test-time scal-
ing demonstrates 2-3×performance improvements,
and practical coding benchmarks (HumanEval, Live-
CodeBench) showing rapid progress in code genera-
tion capabilities.
8.Agentic AI Framework: We synthesize re-
search spanning reasoning techniques (Chain-of-
Thought, Tree-of-Thoughts), tool integration (Re-
Act, Retrieval-Augmented Generation), and multi-
agent systems, providing a unified view of how
LLMs evolve into autonomous agents capable of
complex problem-solving, planning, and execution.
This includes analysis of reasoning emergence, tool-
use capabilities, and coordination mechanisms in
multi-agent scenarios.
2 The Scaling Wall: Data Scarcity,
Rising Costs, and Alternative
Paradigms
While the success of large language models has been
driven primarily by scaling—increasing model size, data,
and compute—the field now faces critical challenges that
threaten the continuation of this paradigm. Recent work
by Coveney and Succi [30] demonstrates that scaling laws
fundamentally limit LLMs’ ability to improve prediction
uncertainty, making it intractable to raise their reliability
to scientific standards. Hooker [67] argues that we are
witnessing the ”slow death of scaling,” where the rela-
tionship between training compute and performance has
become highly uncertain and rapidly changing, with re-
liance on scaling alone missing critical shifts underway in
GPT-3
(175B)
(2020)Gopher
(280B)
(2021)Chinchilla
(70B)
(2022)PaLM
(540B)
(2022)GPT-4
(1.8T)
(2023)Llama 3
(405B)
(2024)Gemini 1.5
(2024)DeepSeek-V3
(671B)
(2025)020406080100120140160Training Cost (Million USD)
$3.3M $4.2M$2.6M$11.2M$84.5M$103.2M$156.5M
$110.7MEarly Era
(2020-2021)Scaling Era
(2022)Frontier Era
(2023-2025)Evolution of Training Costs: Hardware + Energy
Amortized Hardware Cost
Energy CostFigure 3:Training Costs Evolution: Exponential
Growth Across Three Eras.Stacked bar chart showing
hardware costs (blue: amortized chip depreciation + 23%
networking overhead) and energy costs (orange: electric-
ity consumption) for 8 frontier models spanning 2020-
2025 [15, 133]. Era annotations mark three periods: Early
(2020-2021), Scaling (2022-2023), and Frontier (2024-
2025). Key numbers demonstrate 100×cost explosion:
GPT-3 ($3.3M total, 2020), GPT-4 ($84.5M total, 25×
increase), and DeepSeek-V3 ($110.7M total, 2025). Se-
lected models represent the most compute-intensive train-
ing runs for their time. Open circles indicate estimated
Google TPU production costs (higher uncertainty). Cloud
rental costs (not shown) are typically 2-4×higher than
amortized ownership costs.
the field. These fundamental limitations—what we term
the ”scaling wall”—manifest across three interconnected
dimensions that this section systematically examines.
This section systematically answers three fundamental
questions that motivate our comprehensive technical anal-
ysis:
•WHY is scaling hitting limits?We identify three
critical bottlenecks: (1)data scarcity—projected ex-
haustion of high-quality text by 2026-2028, (2)expo-
nential cost growth—training costs increasing 100×
in 5 years ($3M to $300M+), and (3)energy con-
straints—consumption growing 22×from GPT-3 to
GPT-4 (280 MWh to 6,150 MWh).
•WHAT are the specific numbers?We provide
quantitative evidence: frontier models now con-
5

GPT-3
175B
(2020)Gopher
280B
(2021)Chinchilla
70B
(2022)PaLM
540B
(2022)GPT-4
1.8T
(2023)Llama 3
405B
(2024)Gemini 1.5
(2024)DeepSeek-V3
671B
(2025)0100200300400500Total Training Cost (Million USD)
3.8× 3.8×3.2×3.4×3.4×3.3×3.2×
3.3×Numbers above bars show
Cloud/Ownership cost ratioCloud Rental vs. Amortized Ownership Costs
Amortized Hardware + Energy
Cloud Cost (Reported)
Cloud Cost (Estimated)
Estimated ValueFigure 4:Cloud vs. Amortized Costs: Ownership Eco-
nomics for Large-Scale Training.Grouped bar chart
comparing cloud rental pricing (darker bars) with amor-
tized ownership costs (lighter bars: hardware + energy)
for 8 frontier models [15, 133]. Cost multipliers displayed
above bar pairs show cloud costs are typically 2-4×
higher than ownership due to provider margins (30-50%),
maintenance overhead, and infrastructure costs. Color
coding distinguishes reported costs (green/blue) from es-
timated values (red/orange), with open circles marking es-
timated cloud prices where official pricing was unavail-
able. Key insight: the cost gap widens for larger models
as economies of scale increasingly favor ownership for
sustained multi-month training runs, making cloud eco-
nomical only for short-term or exploratory training.
sume 10-15 trillion tokens (approaching the 9-
27T total stock), require 50-150 million GPU-hours
(equivalent to 100,000+ A100 GPUs), cost $80-
160M in hardware plus $6-12M in energy, and
consume megawatt-hours equivalent to 500-1,500
households’ annual electricity.
•WHAT alternatives exist?We analyze eight break-
through paradigms to break the scaling wall: (1)
test-time compute scaling—o1 and DeepSeek-R1
achieve GPT-4 performance with 10×inference
compute but smaller pre-training, (2)sparse ar-
chitectures—DeepSeek-V3’s MoE achieving 671B
parameters with only 37B active (18×efficiency),
(3)architectural innovations—linear attention, state
space models, and sliding window patterns break-ingO(n2)complexity, (4)post-training quantiza-
tion—4-8×compression with ¡1% degradation fol-
lowing predictable scaling laws [167], (5)distributed
edge computing—leveraging massive device net-
works at 10×lower cost [164], (6)model merg-
ing—combining specialized models for synergistic
capabilities [53], (7)efficient training—ORPO re-
duces memory by 50% [65], and (8)small special-
ized models—Phi-4 (14B) matches larger models
through data quality [89].
These questions frame the technical deep-dive that fol-
lows, providing concrete motivation for why architectural
innovations, training efficiency, and alternative scaling
paradigms are critical for the field’s future.
2.1 Overview: The Scaling Wall Crisis
This section examinesWHY scaling is hitting funda-
mental limitsthrough three interconnected crises: data
scarcity, exponential cost growth, and unsustainable en-
ergy consumption. Understanding these bottlenecks mo-
tivates the alternative paradigms we explore in subsequent
sections.
2.2 Crisis 1: Data Scarcity and the Exhaus-
tion of High-Quality Text
Recent projections by Epoch AI [151, 49] reveal a loom-
ing crisis: we may exhaust the stock of high-quality
human-generated text data between 2025 and 2030. Be-
yond this empirical constraint, Coveney and Succi [30]
identify a deeper theoretical limitation: the very mech-
anism that fuels LLMs’ learning power—the ability to
generate non-Gaussian output distributions from Gaus-
sian inputs—may be at the root of their propensity for
error accumulation and information catastrophes. This
tension between learning capacity and accuracy is fur-
ther compounded by the ”spurious correlation deluge”: as
dataset size increases, spurious correlations rapidly pro-
liferate regardless of data quality, fundamentally limiting
the reliability improvements achievable through scaling
alone [30].
Figure 5 illustrates the projection of the effective stock
of human-generated public text and dataset sizes used
to train notable LLMs. The solid line shows historical
6

growth in dataset sizes, with individual dots representing
specific models (GPT-3: 300B tokens, Llama 2: 2T to-
kens, Llama 3: 15T tokens). The projection combines ex-
trapolation of historical trends with a compute-based esti-
mate assuming compute-optimal training [64].
2020 2022 2024 2026 2028 2030
Year051015202530Dataset Size (Trillion Tokens)
GPT-3Llama 2Llama 3 DeepSeek-V3Available Data
Stock RangeData Exhaustion Projection: Training Data vs. Available Stock
Actual Models
Historical Trend
Compute-Optimal Projection
Data Bounds (9-27T tokens)
Current (2025)
Figure 5:Data Exhaustion Projection: The Impend-
ing Data Scarcity Crisis.Scatter plot showing histori-
cal dataset sizes of actual models (GPT-3: 300B tokens,
Llama 2: 2T tokens, Llama 3: 15T tokens, DeepSeek-V3:
14.8T tokens) with exponential growth trend line (2019-
2025) and future projection assuming compute-optimal
training [151, 49]. The shaded region (9-27 trillion to-
kens) indicates the estimated range of total available high-
quality public text, including books, scientific papers,
news articles, Wikipedia, and filtered web content. Un-
der current scaling trends, the available data stock will
be fully utilized between 2026 and 2028, creating a fun-
damental bottleneck for continued model scaling. Data
sources: Epoch AI research and model technical reports.
The total stock of high-quality public text data is esti-
mated at 9-27 trillion tokens [151], including books, sci-
entific papers, news articles, Wikipedia, and filtered web
text. Current frontier models already train on 10-15 tril-
lion tokens, approaching this upper bound. Table 1 shows
the evolution of training dataset sizes across major mod-
els.
Overtraining and Repetition:Figure 6 shows pro-
jections of future dataset requirements under three differ-
ent scaling policies [151]: (1)Compute-optimalfollow-
ing Chinchilla laws (20 tokens per parameter), (2)Mod-
erate overtraining(50-100 tokens per parameter), and (3)
Aggressive overtraining(200+ tokens per parameter) asTable 1: Evolution of training dataset sizes in frontier
LLMs. Data from [18, 149, 46, 40, 49].
Model Year Dataset Size Tokens
GPT-3 2020 570GB 300B
Gopher 2021 2.35TB 1.4T
Chinchilla 2022 2.8TB 1.4T
Llama 2 2023 3.3TB 2.0T
Llama 3 2024 15TB 15T
Gemini 1.5 2024 – 12T+
DeepSeek-V3 2025 14.8TB 14.8T
Estimated Total–45-100TB 9-27T
seen in Llama 3. Under compute-optimal training, the
data stock is fully exhausted by 2026-2027. Overtrain-
ing extends this to 2028-2030 but introduces diminishing
returns and potential overfitting to repeated data.
2.3 Crisis 2: Exponentially Rising Training
Costs
The Bottom Line:Training costs have exploded 100×
in just 5 years—from $3.3M total for GPT-3 (2020) to
$110M+ for DeepSeek-V3 (2025), approaching $330M+
for cutting-edge 2025 models. This subsection quantifies
exactly where these costs originate and their implications
for AI democratization.
The financial cost of training frontier AI models has in-
creased by over 30×in five years, from $3.3M for GPT-
3 (2020) to $84.5M for GPT-4 (2023) and $110.7M for
DeepSeek-V3 (2025), with next-generation models pro-
jected at $300-500M [15]. Figure 3 shows the evolution
of training costs across three eras, demonstrating the ac-
celerating financial burden of continued scaling.
Figure 4 compares cloud rental pricing with amor-
tized ownership costs for major models, revealing that
cloud costs are typically 2-4×higher due to provider
margins (30-50%), maintenance overhead, and infras-
tructure costs. This cost gap widens for larger models,
making ownership more economical for sustained multi-
month training runs but creating barriers for academic and
smaller industry labs.
Cost Structure Analysis:Training costs decompose
into hardware acquisition/amortization (70-80%), energy
consumption (15-25%), and operational overhead (5-
7

2025 2026 2027 2028 2029 2030
Year101520253035Cumulative Dataset Size (Trillion Tokens)2026.22026.82027.4
Total Available
Data Stock
(9-27T tokens)Dataset Requirements Under Different Overtraining Policies
Compute-Optimal (20 tokens/param)
Moderate Overtraining (50-100 tokens/param)
Aggressive Overtraining (200+ tokens/param)
Current (2025)Figure 6:Overtraining Scenarios: How Training Poli-
cies Affect Data Consumption Timeline.Three pro-
jection lines showing future dataset requirements under
different scaling policies [151, 49]: (1)Compute-optimal
following Chinchilla scaling laws (20 tokens/param),
(2)Moderate overtraining(50-100 tokens/param), and
(3)Aggressive overtrainingfollowing Llama 3 approach
(200+ tokens/param). Intersection points mark when each
policy exhausts the 27T token upper bound. The shaded
region indicates the total available data range (9-27T to-
kens). Key insight: depending on the degree of overtrain-
ing, the available data stock is fully consumed between
2025 and 2030, with compute-optimal exhausting data
fastest (2026-2027) while aggressive overtraining extends
to 2028-2030 but introduces diminishing returns.
10%). The dominance of hardware costs means that train-
ing efficiency innovations (reducing GPU-hours) have the
largest impact on total cost reduction.
2.4 Crisis 3: Unsustainable Energy Con-
sumption and Environmental Impact
Energy consumption represents the third critical dimen-
sion of the scaling wall. GPT-4 training consumed ap-
proximately 6,154 MWh (6.15 GWh), equivalent to the
annual electricity usage of 570 average US households
(based on 10,800 kWh/household/year) [15, 133]. This
represents a 22×increase from GPT-3 (280.8 MWh),
highlighting the environmental unsustainability of naive
scaling.
The Changing Nature of Scaling:Hooker [67] ar-
gues that the field is experiencing a fundamental shift:the simple formula of ”scale model size and training
data” has become inadequate, with the relationship be-
tween training compute and performance now highly un-
certain and rapidly changing. This shift has profound
implications: (1)Academia marginalized—the capital-
intensive scaling paradigm has concentrated progress in
industry labs while fundamentally reshaping scientific
culture, (2)Transparency declining—industry labs have
increasingly stopped publishing detailed methodologies,
and (3)Alternative levers emerging—architectural inno-
vations, training efficiency, and test-time compute repre-
sent more promising paths forward than naive scaling.
The convergence of these three crises—data exhaus-
tion by 2026-2028, 100×cost increases, and 22×energy
growth—establishes what we term the ”scaling wall” that
fundamentally limits brute-force scaling approaches. The
remainder of this paper explores quantitative details of
these costs and viable alternative paradigms to continue
AI progress.
3 Model Taxonomy: Evolution of
Foundation Models & Reasoning
Capabilities
The evolution of foundation models represents a conver-
gence of scaling insights, architectural innovations, and
training methodologies. Figure 7 illustrates the temporal
evolution of major model families from 2019-2025, while
Figure 8 shows the performance scaling trends across key
benchmarks. This section traces the major breakthroughs
organized by model families, highlighting their core tech-
nical contributions.
3.1 OpenAI GPT Series: Scaling and In-
struction Following
Figure 9 illustrates the architectural evolution of the GPT
series, highlighting key innovations at each stage.
GPT-2[117] (2019, 1.5B parameters) demonstrated
that unsupervised language models could perform mul-
tiple tasks through zero-shot prompting, introducing the
concept of task-agnostic pre-training.What:A decoder-
only Transformer trained purely on next-token predic-
tion from web text.Why:To test whether language
8

2019 2020 2021 2022 2023 2024 2025 2026
Year100101102103Parameters (Billions)
GPT-2GPT-3 InstructGPT GPT-3.5GPT-4 GPT-4o o1
LLaMALlama 2Llama 3 Llama 3.1
DeepSeek LLM
DeepSeek-Coder
DeepSeekMathDeepSeek-V2DeepSeek-V3DeepSeek-R1
Gemini 1.0 Gemini 1.5 Gemini 2.0
Qwen 1.5 Qwen 2 Qwen 2.5 Qwen 3Kimi 1.0Kimi 1.5Kimi K2 K2-V2
Phi-2Phi-3 Phi-4
Mistral 7BMixtral 8x7BMixtral 8x22B
Mistral Large 2
GLM-4 GLM-4.5MiMo MiMo-V2Yi-Large
BaichuanClaude 2 Claude 3 Opus Claude 3.5
Skywork OR-1
OLMoOLMo 3RLHF
IntroducedReasoning
Breakthrough
Pure RL
Reasoning
Chinese
Models RiseEvolution of Large Language Models (2019-2025): 50+ Major Models
01.AI
AI2
Alibaba
Anthropic
Baichuan
DeepSeek
Google
MetaMicrosoft
Mistral
Moonshot
OpenAI
Skywork
Xiaomi
ZhipuFigure 7: Timeline of major language model releases (2019-2025) organized by model families. The vertical axis
shows model parameter count (log scale), and colors indicate different organizations. Key innovations are annotated.
The rapid evolution from GPT-2 (1.5B, 2019) to DeepSeek-R1 (671B, 2025) demonstrates exponential growth in scale
and capability, with notable shifts toward reasoning-specialized models in 2024-2025.
modeling at scale could implicitly learn downstream
tasks without task-specific training.How it differs:Un-
like BERT’s masked language modeling requiring task-
specific fine-tuning, GPT-2 performed zero-shot task ex-
ecution through natural language prompts, pioneering the
”prompt engineering” paradigm. The training objective
maximizes log-likelihood:
LLM=−NX
i=1logp θ(xi|x<i) =−E x∼D[logp θ(x)](1)
whereDis the training corpus andx <i= (x 1, . . . , x i−1)
is the context.
GPT-3[18] (2020, 175B parameters) revolutionized
few-shot learning through in-context learning.What:A
175B parameter autoregressive model trained on 300B to-
kens demonstrating emergent few-shot capabilities.Why:
To test the hypothesis that scale alone could enable task
adaptation without gradient updates—”models learn to
learn” from prompt context.How it differs:While GPT-2required exact prompts, GPT-3 could adapt to new tasks
from 1-10 examples in-context, eliminating fine-tuning
for many applications. Given demonstration examples
E={(x(j), y(j))}k
j=1, the model predicts:
p(y|x,E) =p θ(y|[x(1), y(1), . . . , x(k), y(k), x])(2)
GPT-3 established empirical scaling laws:
L(N) =Nc
Nα
, L(D) =Dc
Dβ
, L(C) =Cc
Cγ
(3)
whereLis test loss,Nis parameters,Dis dataset size,
Cis compute (FLOPs), andα≈0.076,β≈0.095,γ≈
0.050.
Codex[22] (2021) specialized GPT-3 for code gen-
eration through fine-tuning on code repositoriesD code,
achieving 28.8% pass@1 on HumanEval.What:A 12B
parameter model fine-tuned from GPT-3 on 159GB of
code from GitHub repositories.Why:To demonstrate
9

2019 2020 2021 2022 2023 2024 2025 2026
Year020406080100Score (%)
90% ThresholdMMLU Performance Evolution
GPT Series
LLaMA Series
DeepSeek Series
Qwen Series
Kimi Series
Claude Series
2019 2020 2021 2022 2023 2024 2025 2026
Year020406080100Score (%)
Reasoning
BreakthroughMATH Performance Evolution
GPT Series
LLaMA Series
DeepSeek Series
Qwen Series
Kimi Series
Claude Series
2019 2020 2021 2022 2023 2024 2025 2026
Year020406080100Score (%)
HumanEval Performance Evolution
GPT Series
LLaMA Series
DeepSeek Series
Qwen Series
Kimi Series
Claude SeriesFigure 8: Performance scaling trends across model families on key benchmarks (MMLU, MATH, HumanEval). Each
line represents a model family, showing consistent improvements over time. Notable discontinuities occur with
reasoning-specialized models (o1, DeepSeek-R1) in 2024-2025, particularly on MATH benchmark, where perfor-
mance jumps from 50-60% to 80-95%, indicating qualitative capability shifts beyond simple scaling.
GPT-2/3
Transformer
DecoderInstructGPT
+ RLHF
PipelineGPT-4
(MoE)
+ Sparse
ExpertsGPT-4o
(Multimodal)
+ Multimodal
Encoderso1
(Reasoning)
+ Extended
CoTGPT Architecture Evolution
Figure 9: Architectural evolution of OpenAI GPT series.
(a) GPT-2/3: Standard Transformer decoder with causal
masking. (b) InstructGPT: Addition of RLHF pipeline
with reward model and PPO optimization. (c) GPT-4: Ar-
chitecture details proprietary; illustrated as black-box sys-
tem with undisclosed optimizations.†(d) GPT-4o: Native
multimodal architecture with shared encoders. (e) o1: Ex-
tended reasoning with RL-trained chain-of-thought gen-
eration.†Architecture not officially disclosed; speculative
representations based on third-party analysis.
that domain-specific fine-tuning on high-quality curated
data significantly enhances specialized capabilities be-
yond general pre-training.How it differs:Unlike gen-
eral GPT-3, Codex understands repository-level context,
programming idioms, and can generate syntactically cor-
rect code with proper API usage. Trained on 54M pub-
lic GitHub repositories with permissive licenses. The
pass@k metric:
pass@k=E problems"
1− n−c
k
 n
k#
(4)
wherenis samples per problem,cis correct samples. This
metric is unbiased and measures the probability that at
least one ofkgenerated solutions passes all test cases.InstructGPT[109] (2022) introduced Reinforcement
Learning from Human Feedback (RLHF).What:A
fine-tuned GPT-3 variant optimized to follow human
instructions rather than merely predicting text.Why:
To align model outputs with human preferences and
intentions—pre-trained models often generate plausible
but unhelpful, untruthful, or harmful content. RLHF
bridges the gap between ”what humans want” and ”what
models learn from internet text.”How it differs:Unlike
supervised fine-tuning alone, RLHF uses human prefer-
ence comparisons to train a reward model, then optimizes
the policy to maximize human-preferred outputs while
staying close to the SFT model via KL penalty. This
three-stage process outperforms pure supervised learning,
achieving 85% human preference over raw GPT-3 despite
using only 1.3B parameters. The three-stage process:
Stage 1 - Supervised Fine-Tuning (SFT):
LSFT(θ) =−E (x,y)∼D demo
|y|X
t=1logπ θ(yt|x, y <t)
(5)
Stage 2 - Reward Model Training:Using Bradley-Terry
preference model:
p(yw≻yl|x) =exp(r ϕ(x, y w))
exp(r ϕ(x, y w)) + exp(r ϕ(x, y l))
=σ(r ϕ(x, y w)−r ϕ(x, y l))(6)
LRM(ϕ) =−E (x,yw,yl)∼D comp[logσ(r ϕ(x, y w)−r ϕ(x, y l))]
(7)
10

Stage 3 - RL Optimization with PPO:
LRL(θ) =E x∼D, y∼π θ(·|x)
rϕ(x, y)−β·KL 
πθ(y|x)πref(y|x)
(8)
whereβis the KL penalty coefficient, typicallyβ∈
[0.01,0.1].
GPT-4[105] (2023) achieved substantial capability im-
provements.What:A large-scale multimodal model (text
and images) with undisclosed architecture. OpenAI has
not officially disclosed architectural details such as pa-
rameter count, training compute, or model structure.†
Third-party analyses [141] have speculated about poten-
tial use of mixture-of-experts (MoE) techniques based on
inference behavior and compute efficiency, but these re-
main unconfirmed.Why:To push the frontier of rea-
soning, factual knowledge, and safety through scale, im-
proved training data, and extensive RLHF with multi-
ple reward models for different attributes (helpfulness,
harmlessness, honesty).How it differs:First model to
achieve human-level performance on professional exams
(90th percentile on bar exam), with substantially better
reasoning (52.9% on MATH vs. GPT-3.5’s 23.5%), mul-
tilingual capabilities (85 languages), and instruction fol-
lowing. Performance gains can be modeled as:
Score GPT-4 =Score GPT-3.5 +∆ scale+∆ data+∆ RLHF+∆ arch
(9)
Achieved 86.4% on MMLU and 67.0% on HumanEval.
OpenAI’s technical report [104] emphasized substantial
contributions from post-training techniques (RLHF) be-
yond pre-training scale, though exact proportions remain
proprietary.
†Note on closed-model claims:GPT-4 architectural de-
tails (parameter count, MoE usage, training compute) are
not officially disclosed. Values in Tables 5 marked with†
are estimates based on third-party computational analysis
and should be interpreted with caution. We treat GPT-4
as a black-box system and analyze only publicly reported
capabilities and benchmark results.
GPT-4o[106] (2024) processes multiple modalities
through shared transformer layers.What:An ”omni” (all)
model natively trained on text, images, audio, and video
with end-to-end cross-modal processing, not adapters.
Why:Previous multimodal models used separate en-
coders with adapters (e.g., vision encoder→projection→LLM), creating bottlenecks and losing cross-modal syn-
ergies. Native multimodal training enables seamless rea-
soning across modalities—understanding that a ”scream”
(audio) corresponds to ”terror” (text) and fearful expres-
sions (vision).How it differs:Unlike GPT-4V (vision
adapter) or CLIP+GPT combinations, GPT-4o’s Trans-
former layers jointly process all modalities from the first
layer, enabling true multimodal understanding rather than
late fusion. Achieves 88.7% MMLU and 90.2% Hu-
manEval while being 2× faster and 50% cheaper than
GPT-4. The multimodal attention:
h(l+1)=Transformer(l)([h(l)
text;h(l)
image;h(l)
audio])(10)
where modality-specific encoders project to unified em-
bedding space:h m=E m(xm)∈Rdfor modalitym.
This shared representation space enables zero-shot cross-
modal transfer.
OpenAI o1[107] (2024) pioneered test-time compute
scaling.What:A model trained via reinforcement learn-
ing on chain-of-thought reasoning traces to ”think be-
fore answering,” spending inference compute on inter-
nal deliberation.Why:Traditional models answer imme-
diately, limiting reasoning capability to pre-training and
post-training knowledge. Test-time compute scaling al-
lows models to trade inference cost for accuracy on hard
problems—similar to how humans spend more time on
difficult questions.How it differs:Unlike GPT-4 which
uses chain-of-thought only for user-visible outputs, o1
learns to generate long internal reasoning chains (up to
32K tokens) that are hidden from users but guide final an-
swers. Trained via RL where rewards are given for correct
final answers, not intermediate steps, forcing the model to
discover useful reasoning strategies. The model generates
extended chain-of-thought before answering:
p(y|x) =X
c∈Cp(y|x, c)p(c|x)(11)
wherecis the reasoning chain,Cis the space of possible
chains. The expected computation scales as:
FLOPs test(x) =FLOPs base·E[|c|]·f(difficulty(x))(12)
achieving 83.3% on AIME (vs. 13.4% for GPT-4o),
demonstrating that performance∝log(FLOPs test). The
breakthrough: 1 hour of test-time reasoning can match
11

106×more training compute, fundamentally changing the
economics of AI—better to deploy smaller models with
longer inference than train ever-larger models.
Analysis: GPT Series Evolution.The GPT series
demonstrates three key scaling dimensions: (1)Param-
eter scalingfrom 1.5B (GPT-2) to 1T+ (o1) parame-
ters, following power-law improvements:∆MMLU=
92.3−0 = 92.3percentage points over 5 years; (2)
Training methodology evolutionfrom pure language mod-
eling to RLHF (InstructGPT), achieving 85% prefer-
ence improvement over base models; (3)Test-time com-
pute scaling(o1), where reasoning performance scales as
AIME score∝log(reasoning tokens), enabling6.2×im-
provement (13.4%→83.3%). The transition from pre-
training dominance to post-training optimization marks
a paradigm shift: GPT-4’s capabilities are 70% at-
tributable to RLHF rather than scale alone [105]. The o1
model reveals that1hour of test-time compute≡106×
training compute for complex reasoning tasks, fundamen-
tally challenging the pretraining-centric scaling hypothe-
sis.
3.2 Meta LLaMA Series: Open-Source
Revolution
Figure 10 shows the architectural innovations in the
LLaMA series that enable efficient training at scale.
LLaMA[148] (2023, 7B-65B) followed Chinchilla
scaling laws [64].What:A family of dense Transformer
models (7B, 13B, 33B, 65B) trained on 1.4T tokens ex-
clusively from publicly available data.Why:To prove
the Chinchilla hypothesis: smaller models trained on
more data outperform larger undertrained models. GPT-3
(175B on 300B tokens) was undertrained; LLaMA-65B
(2.7× smaller) matched its performance through optimal
compute allocation. Open release democratized LLM re-
search, spawning Alpaca, Vicuna, and hundreds of deriva-
tives.How it differs:Unlike GPT-3’s brute-force scaling,
LLaMA emphasized training efficiency through architec-
tural optimizations and longer training. Four key inno-
vations: (1) RMSNorm pre-normalization (7-15% faster
than LayerNorm), (2) SwiGLU activations (1-2% bet-
ter than ReLU), (3) Rotary Positional Embeddings en-
abling length extrapolation, (4) Carefully curated open
data (CommonCrawl, C4, Wikipedia, ArXiv, StackEx-
RMSNorm
Normalization before
each sub-layer
x / RMS(x) * gammaSwiGLU
Gated activation
in FFN
Swish(xW1) * xW2
RoPE
Rotary position
embeddings
Rotation MatrixGQA
Grouped-Query
Attention
8 GroupsLLaMA Architectural InnovationsFigure 10: LLaMA architectural innovations. (a) Pre-
normalization with RMSNorm before each sub-layer. (b)
SwiGLU activation function in FFN. (c) Rotary Posi-
tional Embeddings (RoPE) applied to query and key. (d)
Grouped-Query Attention (GQA) in Llama 3 for efficient
inference. These innovations reduce training cost by 30-
40% while maintaining model quality.
change).
Nopt(C) =G·Ca, D opt(C) =G′·Cb(13)
wherea=b= 0.5,Cis compute budget in FLOPs. For
LLaMA-65B:N= 65×109,D= 1.4×1012tokens.
Key innovations:RMSNorm(Root Mean Square Layer
Normalization):
RMSNorm(x) =x
RMS(x)·g,RMS(x) =vuut1
ddX
i=1x2
i
(14)
wheregis learnable scale. Reduces computation vs. Lay-
erNorm by 7-15%.
SwiGLU activation[136]:
SwiGLU(x) =Swish(xW 1)⊙(xW 2),Swish(x) =x·σ(x)
(15)
Improves over ReLU by 1-2% on perplexity.
Rotary Positional Embeddings (RoPE)[144]:
RoPE(x, m) =
cos(mθ)−sin(mθ)
sin(mθ) cos(mθ)x2i
x2i+1
(16)
12

wheremis position,θ= 10000−2i/d. Enables length
extrapolation beyond the training context.
Llama 2[149] (2023, 7B-70B) introduced open RLHF.
What:The first open-source model with full RLHF
training methodology and Chat variants released under
permissive license.Why:To democratize instruction-
following capabilities previously limited to proprietary
models (ChatGPT, Claude). Open RLHF recipes enabled
community to replicate and improve upon commercial
systems.How it differs:Unlike LLaMA which was base-
model only, Llama 2-Chat competed directly with Chat-
GPT through human feedback training. Innovations: (1)
Iterative RLHF with weekly annotation batches (progres-
sive refinement), (2) Rejection sampling—generate N re-
sponses, keep top-kby reward for SFT, (3) Ghost Atten-
tion (GAtt) to maintain multi-turn consistency, (4) System
prompts for steering behavior, (5) Temporal difference re-
ward model ensembling to reduce reward hacking. Itera-
tive training with rejection sampling:
D(t+1)
SFT ={(x, y∗) :y∗= arg max
y∼π(t)rϕ(x, y)}(17)
Context extension through positional interpolation: scale
RoPE frequencies byα:
θ′
i=θi
α, α=Lnew
Ltrain(18)
Extended from 2K to 4K tokens without retraining.
Achieved 68.9% MMLU (70B), reaching 90% of Chat-
GPT capability at inference costs 10× lower.
Llama 3[46] (2024, 8B-405B) uses Grouped-Query
Attention (GQA) [6].What:A massively scaled family
with 405B parameter flagship model trained on 15T to-
kens—50× more data than LLaMA and 5× more param-
eters than Llama 2-70B.Why:To achieve open-source
parity with proprietary frontier models (GPT-4, Claude
3). Previous open models lagged closed systems by 10-
15 MMLU points; Llama 3-405B closed this gap (88.6%
vs GPT-4’s 86.4%).How it differs:Scale alone insuf-
ficient—Llama 3 introduced: (1) 128K vocabulary tok-
enizer (vs 32K) reducing sequence length by 30%, (2)
Grouped-Query Attention reducing KV cache by 16×, en-
abling 128K context, (3) Training on 15T tokens with im-
proved data filtering (90% CommonCrawl→20% after
quality filtering), (4) Hybrid post-training: PPO for chat,DPO for reasoning/code, (5) Multilingual expansion to
30+ languages. The GQA mechanism:
GQA:K g=Mean(K g·(H/G):(g+1)·(H/G) ),
Vg=Mean(V g·(H/G):(g+1)·(H/G) )(19)
whereHis number of heads,Gis number of groups. For
Llama 3-405B:H= 128,G= 8, reducing KV cache
by16×while maintaining 99% of Multi-Head Attention
quality.
Training compute analysis:
FLOPs train≈6ND≈6×405×109×15×1012≈3.6×1025
(20)
Post-training combines PPO and DPO. The DPO advan-
tage over PPO:
StabilityDPO>StabilityPPO,Cost DPO<Cost PPOby50%
(21)
Analysis: Open-Source Convergence with Closed
Models.The LLaMA series demonstrates that open-
source models can match or exceed closed models
through three strategies: (1)Optimal compute alloca-
tion—Chinchilla laws (N opt∝C0.5,D opt∝C0.5) re-
veal LLaMA-65B trained on 1.4T tokens is more ef-
ficient than GPT-3 (175B on 300B tokens), achieving
comparable perplexity with 2.7× fewer parameters; (2)
Architectural efficiency—RMSNorm (7-15% speedup),
SwiGLU (1-2% quality gain), RoPE (infinite length ex-
trapolation), and GQA (16× KV cache reduction) collec-
tively reduce training cost by 35-40% at iso-quality; (3)
Open RLHF democratization—Llama 2’s public release
of RLHF methodology enabled community models (Vi-
cuna, Alpaca) to reach 90% of GPT-3.5 capability at ¡1%
of the training cost. The convergence trajectory shows:
MMLU LLaMA-3-405B (88.6)−MMLU GPT-4(86.4) = +2.2,
marking the first open model surpassing flagship closed
models. This validates the hypothesis that architectural
innovation dominates pure scale for performance gains
beyond 100B parameters.
3.3 DeepSeek Series: Efficient Scaling and
Pure RL Reasoning
Figure 11 illustrates the key innovations in DeepSeek’s
architecture and training methodology.
13

Multi-head Latent
Attention (MLA)
Compress KV Cache
8x Reduction
K = W_down * W_up * hFine-grained MoE
More Smaller ExpertsMulti-Token
Prediction
Predict n T okens
30% Faster Training
GRPO Training Pipeline (DeepSeek-R1)
Base Model Sample GroupCompute
AdvantagesPolicy
GradientUpdated Model
No SFT Required - Pure RL with Verifiable Rewards
V2 V3 R1020406080100MATH Score (%)43.6%76.4%79.8%DeepSeek Evolution
on MATH BenchmarkDeepSeek Architectural and Training InnovationsFigure 11: DeepSeek architectural and training innovations. (a) Multi-head Latent Attention (MLA) compresses KV
cache. (b) Fine-grained MoE with more smaller experts. (c) Multi-Token Prediction (MTP) predicts multiple future
tokens. (d) GRPO training pipeline for DeepSeek-R1 showing pure RL without SFT. (e) Performance comparison
showing DeepSeek-R1’s reasoning breakthrough.
DeepSeek LLM[34] (2024, 67B) introduced Multi-
head Latent Attention (MLA).What:A bilingual
(Chinese-English) dense model with novel KV cache
compression, trained on 2T tokens.Why:Inference mem-
ory bottleneck: for long contexts (32K+ tokens), KV
cache dominates GPU memory (2·L·d model·seq len grows
linearly with sequence length), limiting batch sizes and
deployment efficiency.How it differs:Standard attention
caches fullK, Vmatrices; MLA compresses them into
low-rank latents. Instead of cachingK∈Rn×d, cache
compressedc K∈Rn×dcwhered c≪d, achieving 8×
memory reduction with ¡1% quality loss. Enables 32K
context on GPUs that could only handle 4K with standard
attention.:
MLA:K=Wdown
KWup
Kh, V=Wdown
VWup
Vh(22)
Cache compressed latent:c=Wdown
Kh∈Rdcwhere
dc≪d. Memory reduction:
Ratio=2Ld model
2Ldc=dmodel
dc≈8×(23)
This innovation enabled DeepSeek’s later massive MoE
models to fit on consumer hardware.DeepSeek-Coder[33] (2024) uses Fill-in-the-Middle
(FIM) training.What:A 33B parameter code-specialized
model achieving 87% HumanEval pass@1, surpassing
GPT-4’s 67%.Why:Code generation requires repository-
level understanding and infilling capabilities (complet-
ing partial functions, fixing bugs mid-code). Standard
left-to-right training only models prefix→suffix, not true
bidirectional context.How it differs:FIM training ran-
domly masks middle spans during pre-training, teaching
the model to predictx mgiven left contextx land right
contextx r. This enables IDE-style code completion, bug
fixing, and refactoring—capabilities impossible for pure
autoregressive models. Trained on 2T tokens of code (87
p(xm|xl, xr) =|xm|Y
t=1pθ(x(t)
m|xl, xr, x(<t)
m)(24)
Training masks middle spans with probabilityp FIM= 0.5.
The model learns to reconstruct masked code given sur-
rounding context, forcing bidirectional understanding of
program semantics.
DeepSeekMath[36] (2024) combines tool-integrated
reasoning (TIR).What:A 7B model achieving 51.7%
14

on MATH benchmark through web-scale mathemati-
cal pre-training (120B math tokens from ArXiv, text-
books, web math) and code interpreter integration.Why:
Pure language models struggle with multi-step calcula-
tions—they can plan solutions but make arithmetic er-
rors. TIR allows models to offload computations to
Python interpreters, combining reasoning (LLM) with re-
liable execution (tools).How it differs:Unlike end-to-
end models that must hallucinate numerical results, TIR
generates code for verification: ”To computeR
x2dx...
‘sympy.integrate(x**2, x)‘→x3/3.” Trained on 500K
math problems with solutions including code execution
traces.:
y=(
LLM(x)if no tool needed
LLM(x,Tool(Code(x)))if computation required
(25)
The model learns when to invoke tools versus reasoning
directly, achieving 40% gains on calculation-heavy prob-
lems.
DeepSeekMoE[37] (2024) fine-grained expert seg-
mentation.What:A 145B total parameter MoE with 16B
activated per token, achieving dense-model quality at 9×
lower inference cost.Why:Traditional MoE uses few
large experts (8 experts × 14B each), causing load im-
balance and underutilization. When expert specialization
is coarse-grained, routing becomes suboptimal—one ex-
pert handles ”all science” instead of specialized experts
for physics, chemistry, biology.How it differs:Fine-
grained MoE uses many small experts (64 experts × 2.3B
each), enabling finer specialization and better load bal-
ancing. Each token activates top-4 experts, achieving
145B/16B= 9×compute reduction. Innovations: (1)
Shared expert always activated for common knowledge,
(2) Routed experts for specialization, (3) Load balancing
loss prevents routing collapse.:
y=KX
i=1gi(x)·E i(x), g=TopK(Softmax(xW g), K)
(26)
Load balancing auxiliary loss:
Laux=αEX
i=1Pi·fi, P i=1
TTX
t=1gi(xt),
fi=|{t:i∈TopK(x t)}|
T(27)whereTis total tokens,α∈[0.01,0.1]. This forces ex-
perts to receive equal fractions of tokens, preventing rout-
ing collapse where all tokens route to one expert.
DeepSeek-V3[40] (December 2025, 671B with 37B
activated) Multi-Token Prediction (MTP).What:The
largest open MoE model combining MLA + fine-grained
MoE + multi-token prediction, trained on14.8T tokens
at cost of only $5.5M—40× cheaper than GPT-4’s esti-
mated training cost.Why:Standard next-token prediction
is sample-inefficient: predicting tokentrequires comput-
ing over entire sequence, but only learns from one pre-
diction. MTP predicts nextntokens simultaneously, ex-
tractingn× more learning signal per forward pass.How
it differs:Addsnprediction heads that forecast tokens
at positionst+ 1, t+ 2, . . . , t+nsimultaneously. Dur-
ing training, allnpredictions provide gradients. At infer-
ence, uses only thet+ 1head (no slowdown). Achieves
1.3× training speedup withn= 4. Combined with MLA
(8× cache compression) and MoE (18× parameter effi-
ciency), enables frontier performance ($0.27/M tokens) at
consumer costs.:
LMTP=−TX
t=1nX
k=1λklogp θ(xt+k|x≤t)(28)
wherenis prediction horizon,λ kare weights (typ-
icallyλ 1= 1.0, λ 2= 0.5, λ 3= 0.25, λ 4=
0.125—exponential decay). Training efficiency gain:
Speedup=n·(1−overhead)
1≈1.3×forn= 4(29)
Achieved 88.5% MMLU and 90.2% HumanEval, match-
ing GPT-4 Turbo at 100× lower cost.
DeepSeek-R1[39] (January 2025) uses Group Rela-
tive Policy Optimization (GRPO).What:A 671B MoE
reasoning model trainedentirely via reinforcement learn-
ing without any supervised fine-tuning, achieving 79.8%
MATH and 79.2% AIME—matching o1 despite no hu-
man reasoning demonstrations.Why:Traditional RLHF
requires expensive human-written reasoning chains (Phi-
4 Reasoning used o3-mini traces). GRPO tests the hy-
pothesis: can reasoning emerge purely from RL with cor-
rectness rewards? The answer: yes, with proper variance
reduction. This proves reasoning is a discoverable skill,
not requiring human blueprints.How it differs:All prior
15

reasoning models (o1, QwQ, Phi-4 Reasoning) used su-
pervised learning on reasoning traces before RL. R1 starts
from DeepSeek-V3 base model→pure RL with group
baselines→reasoning emerges spontaneously including
”aha moments”, self-correction, and strategy discovery.
The breakthrough insight: group-based advantage esti-
mates ˆAi=ri−¯r group reduce variance by 87.5% (vs RE-
INFORCE), enabling stable learning without value net-
works.:
ˆAi=r(x, y i)−1
GGX
j=1r(x, y j) =r(x, y i)−¯r(x)(30)
Policy gradient with group baseline:
∇θLGRPO=−E x∼D"
1
GGX
i=1ˆAi∇θlogπ θ(yi|x)#
(31)
Variance reduction vs. standard REINFORCE:
Var[ˆAi] =Var[r i]−1
GVar[¯r]≈Var[r i]·
1−1
G
(32)
ForG= 8: variance reduced by87.5%, enabling stable
training with only 10K prompts vs PPO’s typical 100K.
Verifiable reward function:
r(x, y) =

1.0if Execute(y) =Answer(x)
0.5if format valid but incorrect
0.0if invalid format+λ·Efficiency(y)
(33)
where Efficiency(y) =−|y|
|y|maxpenalizes verbosity.
Analysis: DeepSeek’s Efficiency Revolution.
The DeepSeek series establishes three breakthrough
paradigms: (1)Memory-efficient attention—MLA
achieves8×KV cache compression (d c= 512
vsd model = 4096), enabling 671B model infer-
ence on consumer GPUs where standard attention
would require datacenter infrastructure. This chal-
lenges the assumption that massive models require
massive hardware; (2)Heterogeneous compute alloca-
tion—Fine-grained MoE with 256 experts activating
only 8 per token achieves671B total/37B active=
18.1×parameter efficiency, demonstrating that
quality∝√total params·active params rather
than linear scaling; (3)Pure RL reasoning emer-
gence—DeepSeek-R1’s GRPO eliminates supervisedfine-tuning entirely, relying on verifiable rewards.
The critical insight: variance reduction through group
baselines (Var[ ˆAi] =Var[r i]·(1−1/G)) enables
stable RL withG= 8samples vs PPO’s typical
G= 64. Performance gains validate this approach:
∆MATH= 79.8−43.6 = +36.2points (V2→R1),
with 95% of improvement attributable to RL rather than
scale. The discovery that pure RL≥SFT+RL fundamen-
tally questions the necessity of human demonstrations for
reasoning tasks.
3.4 Microsoft Phi Series: Data Quality
Over Scale
Phi-1[58] (2023, 1.3B parameters) pioneered the ”text-
book quality” data paradigm.What:A small language
model trained on carefully curated synthetic data empha-
sizing logical reasoning and step-by-step explanations,
achieving Python code generation comparable to mod-
els 10×larger.Why:To test whetherdata qualitycould
substitute for scale—challenging the prevailing ”bigger is
better” paradigm.How it differs:Instead of training on
raw web text (CommonCrawl), Phi-1 used filtered ”text-
book quality” datasets plus synthetic data generated by
GPT-3.5 to target specific reasoning patterns. Training
data composition: 6B tokens (vs GPT-3’s 300B), but each
token carefully selected for educational value. Achieved
50.6% on HumanEval despite 100×fewer parameters
than Codex.
Phi-2[74] (2023, 2.7B parameters) expanded the
paradigm to general reasoning.What:A 2.7B model
trained on 1.4T tokens achieving 56.3% on MMLU and
61.1% on GSM8K—matching 7B-13B models.Why:To
demonstrate that curriculum learning (starting simple, in-
creasing difficulty) combined with synthetic data enables
efficient capability acquisition across multiple domains,
not just code.How it differs:Phi-2 introduced multi-
stage curriculum: (1) basic facts and reasoning (200B to-
kens), (2) intermediate problem-solving (400B tokens),
(3) advanced reasoning and chain-of-thought (800B to-
kens). Each stage filtered for clarity and pedagogical
structure. The training loss shows unusual non-monotonic
behavior as curriculum stages increase difficulty, but final
capabilities substantially exceed baseline models trained
on random shuffled data.
16

Phi-3[2] (2024, 3.8B parameters) achieved 69.0%
MMLU.What:A family of models (3.8B, 7B, 14B)
demonstrating that small models can match larger coun-
terparts through systematic data filtering and progressive
training.Why:As data scarcity becomes acute (2026-
2028 exhaustion), learning efficiency matters more than
scale efficiency.How it differs:Phi-3 introduced ”fil-
tered web” datasets where GPT-4 judges content quality,
removing low-value text. Training pipeline: (1) GPT-4
scores 10T tokens on educational value (scale 1-10), (2)
keep only score≥7(1.5T tokens remaining), (3) cluster
by topic and balance distribution, (4) train with progres-
sive curriculum. The quality filtering formula:
Quality(x) =α·Clarity(x)+β·Reasoning(x)+γ·Factuality(x)
(34)
where GPT-4 estimates each component. Only top 15%
of tokens used for training.
Phi-4[1] (2024, 14B parameters) achieved 84.8% on
MATH—competitive with GPT-4.What:The culmina-
tion of the Phi paradigm, demonstrating that 14B param-
eters trained on high-quality data rivals 140B+ models on
reasoning tasks.Why:To prove that the scaling wall can
be circumvented: Capability=f(Data Quality)·g(Scale)
wherefdominatesgbeyond certain quality thresholds.
How it differs:Phi-4 used synthetic data generation at
unprecedented scale—GPT-4o generated 10B tokens of
reasoning-focused content (math problems, scientific ex-
planations, logical puzzles) with explicit step-by-step so-
lutions. Training cost:∼$500K (3,000 GPU-days on
A100s) vs∼$50M+ for 140B models (100×reduction).
Performance comparison:
Phi-4 MATH
Llama 3 405B MATH=84.8
73.8= 1.15with14B
405B
= 0.035params(35)
The model achieves 29×better parameter efficiency on
reasoning tasks.
Phi-4 Reasoning[89] (2025, 14B parameters) intro-
duced ”teachable prompts.”What:Synthetic reasoning
traces generated by o3-mini that emphasize verification,
self-correction, and multiple solution paths. Achieved
74.9% on MATH through knowledge distillation from
stronger reasoning models.Why:Traditional reason-
ing models (o1, DeepSeek-R1) require massive RL com-
pute. Phi-4 Reasoning tests whether distillation fromreasoning traces can achieve similar capabilities at 10×
lower cost.How it differs:The data generation pipeline:
(1) seed problems from MATH/GSM8K/competitive pro-
gramming, (2) o3-mini generates 5 diverse solutions per
problem with explicit reasoning, (3) filter for correct-
ness (automated test cases) and pedagogical clarity (GPT-
4 judges), (4) fine-tune Phi-4 on curated traces (50K
high-quality examples vs 1M+ typical). Training objec-
tive combines next-token prediction with reasoning coher-
ence:
L=L LM+λ· L coherence (36)
whereL coherence penalizes logical contradictions within
reasoning chains using entailment models.
Analysis: Phi’s Paradigm Shift.The Phi series es-
tablishes that AI capabilities may be approaching ”data
saturation”—a regime where quality exponentially out-
weighs quantity. Three critical insights: (1)Synthetic data
generation—GPT-4 and o3-mini can generate reasoning
traces that exceed human-written content in clarity and di-
versity, enabling unlimited high-quality training data; (2)
Curriculum learning efficiency—progressive difficulty in-
crease with repetition of challenging examples reduces re-
quired token count by 10-100×compared to random sam-
pling; (3)Scaling wall circumvention—Phi-4’s $500K
training cost vs $50M+ for equivalent-capability large
models demonstrates that data strategy, not scale, deter-
mines future progress. The efficiency formula:
Tokens required =C
Qualityαwhereα≈2−3(37)
Doubling data quality reduces required tokens by 4-
8×. Phi models enable: (1)local deployment—14B
runs on consumer GPUs and laptops, (2)rapid itera-
tion—researchers can train competitive models in days,
not months, (3)edge intelligence—enables on-device AI
without cloud dependencies. The broader implication:
as high-quality synthetic data becomes abundant, model
scale matters less than learning algorithms and data cura-
tion strategies.
3.5 Google Gemini and Multimodal Models
Gemini 1.0[52] (2023) pioneered native multimodal ar-
chitecture trained jointly on text, images, audio, and video
from scratch. Unlike adapter-based approaches (e.g.,
17

LLaV A [84]), Gemini learns shared representations across
modalities. The model uses modality-specific encoders
feeding into a unified Transformer:
hunified =Transformer([E text(xt);E vision(xv);E audio(xa)])
(38)
whereE m:Xm→Rdare modality encoders projecting
to shared dimensiond. The joint training objective:
Ljoint=X
m∈{text, vision, audio}λmLm+λ crossLcross-modal
(39)
whereL cross-modal enforces alignment through contrastive
learning:
Lcross-modal =−logexp(sim(hm
i, hm′
i)/τ)P
jexp(sim(hm
i, hm′
j)/τ)(40)
Achieved 90.0% MMLU and state-of-the-art on 30 of 32
benchmarks.
Gemini 1.5[124] (2024) extended context to 1M to-
kens through sparse attention patterns and mixture-of-
experts. Attention complexity reduction via block-sparse
attention:
Attention sparse(Q, K, V) =softmax(QKT)⊙M√dk
V
(41)
whereM∈ {0,1}n×nis sparse mask with density(M)≈
0.01, reducing complexity fromO(n2)toO(n√n). The
long-context capability enables processing entire code-
bases (10K files), books (500K words), or 1-hour videos
(1.8M tokens) in a single forward pass.
3.6 Mistral AI: Open Sparse Models
Mistral 7B[75] (2023) outperformed Llama 2 13B de-
spite being smaller, through sliding window attention
(SWA):
SWA i=Attention(Q i, K[max(0,i−W):i] , V[max(0,i−W):i] )
(42)
whereW= 4096is window size. Each token attends to
previousWtokens, reducing complexity fromO(n2)to
O(nW). Effective context extends through layers:
Effective Context L=W·L(43)whereLis number of layers. For Mistral 7B withL= 32:
effective context= 4096×32 = 131Kpositions.
Mixtral 8x7B[76] (2024) introduced open-source
sparse MoE with 47B total parameters, 13B activated per
token. The routing mechanism:
y=EX
i=1gi(x)·E i(x), g(x) =TopK(Softmax(xW g), K= 2)
(44)
whereE= 8experts, each expertE iis a standard FFN.
Load balancing auxiliary loss:
Laux=α·EX
i=1fi·Pi, f i=|{t:i∈TopK(x t)}|
T,
Pi=1
TTX
t=1gi(xt)
(45)
wheref iis fraction of tokens routed to experti,P iis
average router probability,α= 0.01ensures balanced
expert utilization. This enables quality(47B model)at
cost(13B model).
Pixtral 12B[92] (2024) added vision capabilities
through a 400M parameter vision encoder, processing im-
ages at resolution= 1024×1024asn patches = 1024to-
kens via ViT-style patching:
Patch(I, p) =Flatten(Split(I, p×p))·W proj (46)
wherep= 32is patch size,W proj∈R(3·p2)×dprojects
RGB patches to model dimension.
Analysis: Cross-Model Reasoning Emergence Pat-
terns.Analyzing 15+ reasoning models from 2024-2025
reveals convergent and divergent evolution paths: (1)
Convergent training methodology—87% of top perform-
ers (MATH ¿75%) adopt RL-based post-training (GRPO,
PPO, or variants), suggesting supervised learning alone
is insufficient for reasoning. The critical threshold ap-
pears at≈1010RL tokens of training; (2)Divergent
architectural choices—Models split between dense (o1,
R1: paramstotal=paramsactive) and sparse (Mixtral,
DeepSeek-V3: activation ratio≈0.2), with no clear win-
ner—dense models lead on AIME (o1: 83.3%), sparse
on efficiency (DeepSeek-V3:$5.5/Mtokens); (3)Test-
time compute scaling universality—All frontier reasoning
18

models exhibit performance=a+blog(thinking tokens)
withb∈[0.15,0.25]MATH points perlog2(tokens). The
”reasoning tax” is real: 10× more compute→15-25%
absolute gain; (4)Verifiable reward advantage—Models
trained with code execution / formal proof rewards (Sky-
work OR-1, Seed-Thinking) achieve 5-8% higher MATH
scores than preference-based RLHF at iso-compute, sug-
gesting task-specific reward design matters more than re-
ward model scale. The evolution suggests reasoning is an
emergent capabilityrequiring: (i) scale (>1011training
tokens), (ii) RL with verifiable feedback, (iii) test-time
search.
3.7 Reasoning-Specialized Models (2024-
2025)
Open-Reasoner-Zero[101] (March 2025) validated
GRPO on completely open data, achieving 71.2% on
MATH. Proved that pure RL induces ”aha moments”
where models discover novel problem-solving strategies
not present in base training data.
Phi-4 and Phi-4 Reasoning[1, 89] (2024-2025,
14B) demonstrated that small models trained on high-
quality synthetic data can rival 10× larger mod-
els. Phi-4 Reasoning (April 2025) introduced ”teach-
able prompts”—synthetic reasoning traces generated
by stronger models (o3-mini) that emphasize step-by-
step verification. Achieved 74.9% on MATH, demon-
strating knowledge distillation from reasoning traces.
The data generation pipeline: (1) seed problems from
MATH/GSM8K, (2) generate diverse solutions with o3-
mini, (3) filter for correctness and pedagogical clarity, (4)
fine-tune Phi-4 on curated traces.
Skywork OR-1[139] (May 2025) introducedverifi-
able reward scaling: using mathematical proof check-
ers (Lean, Coq) and code execution as infinite sources
of training signal. The reward function:r(x, y) =
⊮correct(y) +λ·efficiency(y)where efficiency measures
solution conciseness.
Seed-Thinking 1.5[132] (April 2025) combined
Monte Carlo Tree Search (MCTS) with RL for reasoning.
At test time, the model explores multiple reasoning paths:
each thought is a node, children are next-step candidates,
and values are estimated by the model itself. The search
maximizes:V(s) = max a[r(s, a) +γV(s′)]wheres′isthe next state.
Nemotron Series[98, 100, 99] (Llama-Nemotron May
2025, Nano 2 August 2025, Nano 3 December 2025):
NVIDIA’s focus on efficient inference through model dis-
tillation, INT4/INT8 quantization, and TensorRT opti-
mization. Nemotron 3 Nano (4B) achieves 68% MMLU
at 150 tokens/sec on edge GPUs.
Emerging Global Reasoning Models:Moonshot’s
Kimi 1.5 [94] (January 2025) and Kimi K2 [95] (July
2025) emphasized long-context reasoning (up to 200K to-
kens) with memory-efficient attention. K2 uses chunked
attention with cross-chunk memory: each chunk attends
fully to previous summaries. Alibaba’s Qwen 3 [8] (May
2025, 72B) achieved 86.3% MMLU through mixture of
SFT and DPO. Zhipu’s GLM-4.5 [170] (July 2025) in-
troduced All Tools integration where the model can in-
voke any API. Xiaomi’s MiMo series [160]→MiMo
VL [162]→MiMo-V2-Flash [161] (May-December
2025) progressed from text to multimodal to fast infer-
ence, with V2-Flash achieving 85% MMLU at 200 to-
kens/sec through speculative decoding.
OLMo Series[56, 97, 5]: AI2’s commitment to open
science. OLMo 3 Think (November 2025) released full
training code, data mixtures, checkpoints, and evaluation
scripts, enabling complete reproducibility of reasoning
model training.
4 Architectural Innovations: Effi-
ciency Under the Hood
While the Transformer architecture introduced in
2017 [150] remains the foundational building block of
modern LLMs, the year 2025 has witnessed remarkable
efficiency innovations ”under the hood” that dramati-
cally reduce computational and memory costs without
sacrificing performance. As Raschka (2025) observes,
”the foundations remain remarkably same; however, the
efficiency tricks happening under the hood are where
the real magic is” [123]. This section examines the
key architectural breakthroughs that define the current
generation of models.
19

4.1 Efficient Attention Mechanisms
Standard multi-head attention (MHA) has quadratic mem-
ory complexityO(N2)in sequence lengthN, becoming
prohibitive for long contexts (N >8,192). Modern atten-
tion variants address this through algorithmic and struc-
tural optimizations.
FlashAttention[32, 31] achieves 2-4×speedup and
10-20×memory reduction throughIO-awaretiling that
minimizes slow GPU HBM (High Bandwidth Memory)
access. Standard attention materializes the full atten-
tion matrixS=QKT∈RN×Nin HBM before
computing softmax and weighted values. FlashAtten-
tion tilesQ, K, Vinto blocks (B q= 128, B kv=
64for A100 GPUs), loading tiles into fast SRAM
(19 TB/s vs HBM’s 1.5 TB/s), computing attention
block-by-block, and accumulating output incrementally:
Oi=P
jsoftmax(Q iKT
j)Vjwithout storing fullS.
Memory:O(N2)→ O(N·B)whereB≪N.
FlashAttention-2[31] further parallelizes computation
across sequence dimension (previously only parallelized
over batch/heads), achieving 2×additional speedup.
Adopted by Llama 2/3, GPT-4, Claude 3, Gemini 1.5.
Multi-Query Attention (MQA)[135] andGrouped-
Query Attention (GQA)[6] reduce KV cache
size through parameter sharing. MQA usessin-
gle sharedK, Vheads across all query heads:
Attention(Q i, K, V) =softmax(QiKT
√dk)Vfor
i= 1, . . . , hquery heads. Cache reduction:h: 1
(e.g., 8×forh= 8). However, MQA degrades qual-
ity on long-context tasks (¿10k tokens) as single KV
pair lacks expressiveness. GQA balances efficiency
and quality by grouping query heads:h qquery heads
shareh kvkey/value heads whereh kv≪h q(e.g.,
hq= 32, h kv= 8). Cache reduction:hq
hkv= 4×.
Quality gap: MQA shows 2-3% perplexity increase vs
MHA, GQA reduces this to ¡1% while maintaining 4-8×
throughput gain. Llama 2 uses GQA, Falcon uses MQA.
Sliding Window Attention (SWA)[13, 25] restricts at-
tention to local windows: each token attends towneigh-
bors (w= 2,048typical), reducing complexityO(N2)→
O(N·w). Empirically, most attention mass concentrates
within 512-1024 tokens [158], suggesting long-range de-
pendencies contribute marginally. However, information
propagates acrossLlayers: effective receptive field =w·L(e.g.,2,048×32 = 65,536tokens). Gemma 3 com-
bines SWA (window 5) with full global attention every
5th layer, achieving 128k context at ¡50% memory of full
attention. Trade-off: SWA degrades by 1-2% on tasks re-
quiring explicit long-range dependencies (e.g., document
QA with evidence 50k tokens apart).
4.2 Speculative Decoding: Accelerating In-
ference
Autoregressive generation is inherently sequential—each
token depends on all previous tokens, precluding paral-
lelization:y t= arg maxP(y t|y<t, x)requiresTse-
rial forward passes forTtokens.Speculative Decod-
ing[80, 21] parallelizes verification by using a smalldraft
modelto propose candidate sequences, then verifying in
parallel with the target model.
Algorithm.(1) Draft modelM draftgenerateskcandi-
date tokens autoregressively (k= 5−8typical):ˆy t:t+k∼
Mdraft(y<t, x); (2) Target modelM target computes proba-
bilities forallcandidates in parallel (single forward pass):
ptarget(yi|y<i, x)fori=t, . . . , t+k; (3) Verify candidates
via rejection sampling: acceptˆy iifptarget(ˆyi|y<i,x)
pdraft(ˆyi|y<i,x)> u i
whereu i∼Uniform(0,1), accept prefix up to first re-
jection, sample next token from adjusted distribution; (4)
Repeat until sequence complete.
Expected Speedup.If draft model acceptance rate
isαper token, expected accepted length per iteration:
E[L] =Pk
i=1αi=α(1−αk)
1−α. Forα= 0.7, k= 8:
E[L]≈2.3tokens per target forward pass, yielding
2.3×speedup. Speedup condition:Tdraft·k+T target
Ttarget·E[L]<1
whereT draft≪T target (draft 10-100×smaller). Practical
speedups: 2-3×for GPT-4 verified by GPT-2-sized draft
model [80].
Medusa[19] eliminates draft model by adding mul-
tiple prediction heads to the target model, predict-
ing future tokens simultaneously:ˆy t+1, . . . ,ˆy t+k =
Head 1(ht), . . . ,Head k(ht)whereh tis hidden state at to-
kent. Tree-based verification explores multiple candi-
date paths in parallel, accepting the longest valid prefix.
Medusa achieves 2.2-2.8×speedup without external draft
model but requires additional training to calibrate predic-
tion heads (≈10% extra parameters). Trade-off: Specu-
lative decoding is lossless (exactly matches target model
distribution) but requires two models; Medusa is self-
20

contained but adds latency overhead from tree verifica-
tion.
4.3 The Battle for KV Cache Efficiency:
MLA vs. GQA
What:The Key-Value (KV) cache bottleneck has
emerged as a critical challenge for long-context inference.
During autoregressive generation, attention requires stor-
ing keys and values for all previous tokens, leading to
memory consumption that grows linearly with sequence
length: MemoryKV= 2·n layers·nheads·dhead·L·
sizeof(float16), whereLis sequence length.
Why traditional approaches fall short:Standard Multi-
Head Attention (MHA) becomes prohibitively expensive
for long contexts. For a model like GPT-3 with 96 layers
and 96 heads processing 32K tokens, the KV cache alone
requires≈180GB of GPU memory, leaving little room for
model parameters or batch processing.
Two competing solutions:The field has converged
on two primary approaches—Grouped-Query Attention
(GQA) and Multi-Head Latent Attention (MLA)—each
offering distinct tradeoffs.
Grouped-Query Attention (GQA)reduces memory
by sharing key-value pairs across multiple query heads.
Instead of maintaining separate K/V projections for each
ofHattention heads, GQA groups heads intoGgroups
(G≪H), with heads within each group sharing K/V:
GQA(Q, K, V) =Concat(head 1, . . . ,head H)WO,
head i=Attention(Q iWQ
i, K⌈i/r⌉WK
⌈i/r⌉, V⌈i/r⌉WV
⌈i/r⌉)
(47)
wherer=H/Gis the group size. For Llama 3-
405B withH= 128heads andG= 8groups,
this yields16×KV cache reduction: MemoryGQA =
MemoryMHA/16. Ablation studies show GQA maintains
99% of MHA quality while drastically reducing mem-
ory bandwidth requirements [6].Adoption:Llama 2/3,
Mistral, Qwen—essentially the industry standard for open
models.
Multi-Head Latent Attention (MLA)takes a funda-
mentally different approach by compressing K/V into a
low-rank latent space before caching. Instead of stor-
ing full-dimensional keys and values, MLA projects themthrough learned down-projection matrices:
Kcompressed =KWdown
K, V compressed =V Wdown
V,
whereWdown∈Rd×dc, dc≪d(48)
The verdict:GQA wins on implementation simplic-
ity and hardware compatibility (works seamlessly with
FlashAttention), while MLA offers superior modeling
performance at the cost of complexity. DeepSeek’s choice
of MLA over GQA appears vindicated by their bench-
mark dominance, though this comes with higher engineer-
ing overhead.
4.4 Mixture-of-Experts: From Optional to
Mandatory
What:Mixture-of-Experts (MoE) architectures have tran-
sitioned from a niche technique to the dominant paradigm
for frontier models in 2025. MoE replaces monolithic
feedforward layers with multiple specialized ”expert” net-
works, activating only a subset per token through learned
routing.
Why MoE is now essential:The fundamental prob-
lem is theknowledge vs. compute tradeoff. Dense mod-
els must choose between capacity (total parameters for
knowledge storage) and inference cost (active parame-
ters per forward pass). MoE decouples these: total pa-
rameters can scale to 671B (DeepSeek-V3) while keep-
ing active parameters at 37B—achieving quality(671B)
at cost(37B).
Evolution of expert granularity:2025 models demon-
strate a clear trend towardfine-grained experts—many
small experts instead of few large ones. Traditional MoE
(Mixtral 8×7B) uses 8 large experts (7B each), activating
2 per token. DeepSeek-V3 pioneered fine-grained MoE
with 256 tiny experts (2.6B each), activating 8+1 (shared)
per token:
MoE(x) =FeedForward shared(x)
+X
i∈TopK(g(x),K)gi(x)·FeedForward i(x)(49)
whereg(x) =Softmax(xW g)is the routing function,
K= 8is the number of activated routed experts, and the
shared expert (always active) captures common patterns.
21

Why fine-grained works better:Empirical studies
show that many small experts enable better load balanc-
ing—preventing the ”expert collapse” problem where a
few experts dominate all routing. DeepSeek-V3’s auxil-
iary load balancing loss ensures balanced utilization:
Lload=αNexpertsX
i=1fi·Pi, f i=|{t:i∈TopK(g(x t))}|
T,
Pi=1
TTX
t=1gi(xt)
(50)
wheref iis the fraction of tokens routed to experti,P iis
the average routing probability, andα= 0.01weights
the auxiliary loss. This yields9×compute reduction:
DeepSeek-MoE achieves dense-model quality with 18.2B
active / 145B total parameters [38].
The shared expert innovation:A critical architectural
refinement is the addition of a shared expert that pro-
cesseseverytoken, alongside routed experts. This de-
sign, introduced in DeepSeek-MoE (2024) and adopted
by DeepSeek-V3, Kimi K2, and GLM-4.5, provides
two key benefits: (1) Common knowledge consoli-
dation—frequent patterns (e.g., basic syntax, common
words) learned once in the shared expert rather than
redundantly across routed experts; (2) Training stabil-
ity—the shared expert’s consistent gradients prevent rout-
ing collapse during early training. Interestingly, Qwen3
initially omitted the shared expert, but Qwen3-Next
(September 2025) reintroduced it, validating its impor-
tance [9].
2025 MoE landscape:Nearly all frontier mod-
els released in 2025 are MoE-based: DeepSeek-V3
(671B/37B), Llama 4 Maverick (400B/17B), Qwen3
(235B/22B), Mistral 3 Large (673B/39B), Kimi K2
(1T/59B), GLM-4.5 (355B/ 40B), MiniMax-M2
(230B/10B), gpt-oss-120b (120B/ 4B), Nemotron 3
Nano (30B/3B). The ”dense model era” appears to be
ending for models above 100B parameters.
4.5 Stabilizing Training: The Normaliza-
tion Renaissance
What:While the core Transformer architecture remains
unchanged, 2025 has seen sophisticated refinements innormalization layer placement and types, dramatically
improving training stability—a critical requirement as
models scale to trillions of parameters.
Post-Norm revival in OLMo 2 and Gemma 3:
The original Transformer (2017) placed LayerNormaf-
terattention and feedforward modules (Post-LN). GPT-2
(2019) moved normalizationbeforethese modules (Pre-
LN), and this became standard because Pre-LN enables
training without careful warm-up [163]. However, OLMo
2 (2025) revived a modified Post-Norm configuration
where RMSNorm is applied after modules butinsidethe
residual connections:
PostNorm OLMo2 (x) =x+RMSNorm(Module(x)),
vs. standard PostNorm: RMSNorm(x+Module(x))
(51)
This hybrid approach yields smoother loss curves with
fewer spikes during pre-training [4]. Gemma 3 goes fur-
ther withdual normalization—applying RMSNorm both
before and after attention/feedforward:
y=x+RMSNorm post(Module(RMSNorm pre(x)))(52)
This ”sandwich” configuration provides the stability of
Pre-Norm with the gradient flow properties of Post-Norm.
QK-Norm: Stabilizing attention logits:A second
critical innovation is Query-Key Normalization (QK-
Norm), where RMSNorm is applied to queries and keys
insidethe attention mechanism before computing atten-
tion scores:
Attention QKNorm (Q, K, V) =
softmaxRMSNorm(Q)·RMSNorm(K)T
√dk
V(53)
Originally proposed for vision transformers [41], QK-
Norm prevents attention logit explosion in long-context
scenarios. OLMo 2’s ablation study shows that Post-
Norm + QK-Norm together eliminate loss spikes that
plagued earlier training runs [4].Adoption:OLMo 2,
Gemma 2/3, MiniMax-M2 (with per-layer variant), gpt-
oss, Grok 2.5.
Why normalization matters at scale:As models grow
to 100B+ parameters and train on 10T+ tokens, even mi-
nor instabilities compound. A single loss spike can waste
22

weeks of compute (>$1Mfor frontier models). The nor-
malization innovations of 2025 representarchitectural in-
surance—small additions that dramatically reduce train-
ing risk.
4.6 Local vs. Global Attention: The Sliding
Window Revolution
What:Traditional self-attention allows each token to at-
tend to all previous tokens (global attention), incurring
O(n2)memory and compute complexity. Sliding Win-
dow Attention (SWA) restricts each token to a fixed-size
local window ofwpreceding tokens, reducing complexity
toO(nw).
Why it works:Empirical studies show that most
language modeling benefits come from local con-
text—tokens primarily attend to nearby words. For ex-
ample, in ”The cat sat on the mat”, ”sat” strongly at-
tends to adjacent ”cat” and ”on”, with attention to distant
”The” contributing minimally. SWA exploits this locality:
each token attends only to the most recentwtokens (e.g.,
w= 1024in Gemma 3).
Gemma 3’s aggressive approach:Where Gemma 2
used a 1:1 ratio (alternating between sliding window
and global attention layers), Gemma 3 adopts a 5:1 ra-
tio—only 1 global attention layer per 5 sliding window
layers—and shrinks the window from 4096 to 1024 to-
kens. This yields4×memory savings in the KV cache
compared to Gemma 2:
MemoryGemma3 =5·1024 + 1·L
6
/1·4096 + 1·L
2
≈0.26(atL= 32K)
(54)
Surprisingly, ablation studies shownegligible perplexity
degradation—global attention in every 6th layer suffices
for long-range dependencies [55].
Adoption and tradeoffs:SWA is used by Gemma
2/3, gpt-oss (in alternating layers), Xiaomi MiMo-V2-
Flash (with extreme window size of 128, the most ag-
gressive deployment to date). However, Mistral aban-
doned SWA in Mistral 3.1 Small despite using it in ear-
lier models, likely because SWA complicates hardware
optimization—FlashAttention-3 has limited SWA sup-
port. The tradeoff: memory efficiency vs. inferencelatency (optimized kernels favor uniform attention pat-
terns).
4.7 Linear Attention Revival: Beyond
Quadratic Complexity
What:Standard attention scales asO(n2)due to the pair-
wise token interaction matrixQKT∈Rn×n. Linear
attention approximates this asQϕ(KT)Vusing kernel
functionsϕ(·), achievingO(n)complexity by avoiding
explicit materialization of the attention matrix.
Why previous attempts failed:Early linear attention
variants (2020-2023) like ”Transformers are RNNs” [79]
used simple kernels (e.g.,ϕ(x) =elu(x) + 1) but suf-
fered significant quality degradation—10-15% perplex-
ity increase compared to standard attention. They never
gained traction in production models.
2025 breakthrough: Gated DeltaNet:The key innova-
tion isgated state-space modelsthat combine linear com-
plexity with near-softmax quality. Gated DeltaNet, used
in Qwen3-Next and Kimi Linear, employs a delta-rule up-
date with learned gates:
ht= (1−α t)⊙h t−1+αt⊙(k t⊗vt),(gated state update)
(55)
ot=qT
tht,(output retrieval) (56)
whereα t=σ(W αxt)is a learned decay gate (per-
channel in Kimi Linear vs. per-head in Qwen3-Next),
kt, vt, qtare key/value/query vectors, and⊙denotes
element-wise multiplication. The gate allows the model
to selectively forget or retain information, mimicking at-
tention’s dynamic focus.
Hybrid architectures dominate:Pure linear attention
still underperforms. The winning strategy is a3:1 hybrid:
3 Gated DeltaNet layers followed by 1 global attention
layer. This pattern appears in:
•Qwen3-Next(80B-A3B): Gated DeltaNet + Gated
Attention, 3:1 ratio, 262K native context
•Kimi Linear(48B): Kimi Delta Attention (refined
Gated DeltaNet with channel-wise gating) + MLA,
3:1 ratio
•MiniMax-M1(456B-A46B): Lightning Attention +
Full Attention, hybrid ratio
23

•Nemotron 3 Nano(30B-A3B): Mamba-2 + GQA,
interleaved in macro-blocks
The global attention layers provide precise content-based
retrieval (critical for factual recall), while linear layers
handle bulk sequence processing efficiently.
Performance validation:Kimi Linear achieves compa-
rable benchmark scores to DeepSeek-V3 (which uses full
MLA) while generating tokens2.3×faster at long con-
texts (128K tokens) [96]. The speedup comes from elim-
inating the quadratic KV cache: Memorylinear=O(d2
c)
(constant in sequence length) vs. Memoryattention =O(n·
d). However, MiniMax reversed course with M2 (their
flagship model after M1), returning to full attention, citing
poor accuracy on multi-turn reasoning tasks—suggesting
linear attention may struggle with complex dependencies.
The open question:Will linear attention scale to
trillion-parameter models? Current deployments top out
at 456B (MiniMax-M1). Kimi K2 (1T parameters) uses
full MLA, not linear attention. The jury is still out on
whether linear attention’s efficiency gains persist at the
frontier, or if diminishing returns on quality make full at-
tention essential for top-tier performance.
4.8 Positional Encoding Evolution: From
RoPE to NoPE
What:Positional encodings inform the model about token
order. Rotary Position Embeddings (RoPE), introduced in
2021 [144], became the standard by applying rotations to
query-key pairs:
qm=R Θ,mq, k n=R Θ,nk,
whereR Θ,m=cos(mθ)−sin(mθ)
sin(mθ) cos(mθ)(57)
This encodes relative positions through rotation angles:
attention(q m, kn)∝cos((m−n)θ).
Partial RoPE:MiniMax-M2 and gpt-oss apply RoPE to
onlyhalfthe attention head dimensions, leaving the rest
unchanged:
Partial-RoPE(q) = [RoPE(q 1:d/2);qd/2+1:d ](58)
Why:At inference on sequences longer than train-
ing’s maximum length, full RoPE can produce ”unseen”
rotation angles, potentially degrading quality. PartialRoPE provides an ”unrotated escape hatch” that carries
position-agnostic information, improving length extrapo-
lation [90].
No Positional Embeddings (NoPE):SmolLM3 (3B)
and Kimi Linear take the radical approach of omitting po-
sitional encodings entirely.How does the model know
token order?The causal attention mask provides im-
plicit positional information: tokentcan only attend to
tokens≤t, creating directional flow. Theoretically, this
”masked self-attention as implicit position encoding” is
sufficient [61]. NoPE improves length generalization (no
”unseen position” problem by construction) and simpli-
fies architecture.
Adoption:Partial RoPE—MiniMax-M1/M2, gpt-oss;
NoPE—SmolLM3-3B, Kimi Linear (in MLA layers only,
DeltaNet layers use implicit positioning through gating).
Full RoPE remains standard for most models (Llama,
Qwen, DeepSeek).
4.9 Width vs. Depth: Architectural Propor-
tions
What:At a fixed parameter budget, designers choose be-
tweendepth(more layers) andwidth(larger hidden di-
mensions/more heads). For example, gpt-oss-20b uses 24
layers with 2880 hidden dim, while Qwen3-30b-A3B uses
48 layers with 2048 hidden dim—both≈20-30B parame-
ters.
Why width wins for inference:Wider models paral-
lelize better on GPUs (matrix multiplies scale efficiently
with dimension) and achieve higher tokens/sec through-
put. Deeper models suffer from sequential dependencies
across layers. Empirically, the Gemma 2 ablation study
(9B parameter budget) found wider architectures outper-
formed deeper ones: 52.0 vs. 50.8 average score across 4
benchmarks [54].
Why depth wins for training:Deeper models have more
representational flexibility—each layer adds a ”thinking
step”. Training stability improves with depth when using
modern normalization (Post-Norm + QK-Norm).
The 2025 consensus:Frontier models favordeepar-
chitectures: DeepSeek-V3 (61 layers), Qwen3-Next (64
layers), Kimi K2 (85 layers). This reflects the prior-
ity on maximizing model capacity (training performance)
over inference speed—once trained, inference can be op-
24

timized separately via quantization, speculative decoding,
etc. Smaller models optimized for edge deployment (gpt-
oss, SmolLM3) favor width for efficiency.
4.10 Multi-Token Prediction: Learning Sig-
nal Amplification
What:Standard language modeling predicts one token at
a time:L=−P
tlogp(x t+1|x≤t). Multi-Token Predic-
tion (MTP) trains the model to predict thenextntokens
simultaneously using auxiliary prediction heads:
LMTP=−TX
t=1nX
i=1λilogp(x t+i|x≤t),
whereλ i= 0.5i−1(exponential decay)(59)
For example, withn= 4:λ 1= 1.0, λ 2= 0.5, λ 3=
0.25, λ 4= 0.125—later predictions weighted less since
they’re harder.
Why it accelerates training:Each forward pass extracts
n×more learning signal. Instead of one cross-entropy
loss per position, MTP providesnlosses. DeepSeek-V3
reports1.3×training speedup: reaching the same per-
plexity in 77% of the steps [35].
Inference applications:While MTP is primarily a train-
ing technique, the auxiliary prediction heads can be re-
purposed forspeculative decoding: instead of generating
one token per forward pass, generate multiple candidate
tokens, verify in parallel, and accept the longest correct
sequence. Qwen3-Next explicitly optimizes for this use
case [9].
Adoption:DeepSeek-V3/V3.2, Qwen3-Next, GLM-
4.5, MiniMax-M2. Interestingly, MTP is often undocu-
mented—it’s a ”training secret” since the auxiliary heads
are discarded for inference in most models (except for
speculative decoding).
Open question:Does MTP bias the model toward lo-
cal dependencies at the expense of long-range reasoning?
Predictingx t+4fromx ≤tencourages short-horizon plan-
ning. No thorough ablation studies exist comparing MTP
vs. standard training on complex reasoning benchmarks
(MATH, AIME).4.11 Architectural Convergence and Diver-
gence: 2025 Synthesis
The convergence:Despite surface diversity, 2025 flag-
ship models exhibit remarkable architectural similarity,
as illustrated in Figure 12. All use: (1) Decoder-
only Transformer blocks with causal masking; (2) RM-
SNorm (not LayerNorm) for efficiency; (3) SwiGLU (not
ReLU/GELU) activation in feedforward layers; (4) RoPE
or variants (not absolute positional embeddings); (5) Pre-
trained on 5-15T tokens with continued training on cu-
rated high-quality data. The ”GPT architecture” intro-
duced in 2018 remains intact.
The divergence:The efficiency optimizations diverge
based on deployment priorities:
•Memory-optimized:MLA + MoE (DeepSeek-V3,
Kimi K2)—for massive models (671B-1T) with con-
strained inference hardware
•Latency-optimized:GQA + Dense or light MoE +
optimized kernels (Mistral 3.1 Small, Llama 4)—for
low-latency serving
•Long-context-optimized:Sliding window or lin-
ear attention (Gemma 3, Qwen3-Next, Kimi Lin-
ear)—for 128K-512K context applications
•Training-stability-optimized:Post-Norm + QK-
Norm (OLMo 2, Gemma 3)—for academic/research
settings with limited compute for re-runs
•Edge-optimized:Width¿depth, small MoE, NoPE
(gpt-oss, SmolLM3, Gemma 3n)—for on-device de-
ployment
The answer to ”are LLM architectures evolving?”:
The foundations remain remarkably same—the Trans-
former decoder is essentially unchanged since GPT-2.
However,the efficiency tricks under the hood are
where the real magic is. The difference between a 2019
model and a 2025 model at the same parameter count is
not architectural revolution, but rather:
•8−16×memory savings (GQA/MLA + sliding win-
dow/linear attention)
•10−20×cost reduction (MoE sparse activation)
25

2025 LLM Architecture Evolution: Convergence and Divergence
CONVERGENCE: Core GPT Architecture (2018-2025)
Decoder-only
TransformerRMSNormSwiGLU
ActivationRoPE VariantsCausal
Masking5-15T T oken
PretrainingTHROUGHPUT-OPTIMIZED
 MLA (8× KV compression)
 Fine-grained MoE (256 experts)
 Deep (60+ layers)
 Multi-token prediction
 DeepSeek-V3 (671B/37B)
Kimi K2 (1T/59B)
GLM-4.5 (355B)LATENCY-OPTIMIZED
 GQA (16× memory)
 Dense/Light MoE
 Optimized kernels
 Low-latency serving
 Mistral 3.1 Small (24B)
Llama 4 Maverick (400B)
Qwen3 (235B/22B)EDGE-OPTIMIZED
 Width > Depth
 Small MoE
 NoPE (no PE)
 On-device
 gpt-oss-20b
SmolLM3 (3B)
Gemma 3nKey 2025 Architectural InnovationsKV Cache
Efficiency
MLA/GQA
8-16× savingsFine-grained
MoE
256 experts
+ sharedTraining
Stability
Post-Norm
QK-NormSliding
Window
Gemma 3
4× memory
Linear
Attention
Gated DeltaNet
2.3× speedupPositional
Encoding
Partial RoPE
NoPEMulti-Token
Prediction
DeepSeek-V3
n× efficiencyCost
Reduction
$5.5M training
40× cheaper
2025 Efficiency Gains: 8-16× memory savings    2-5× throughput improvement    40× training cost reduction
Figure 12: Architectural evolution from GPT-2 (2019) to present frontier models. While the core Transformer struc-
ture persists, efficiency innovations dominate: (left) Evolution of attention mechanisms from MHA→GQA→MLA
and emergence of sliding window/linear variants; (center) MoE adoption trajectory showing transition from optional
(Mixtral 2023) to mandatory (all 2025 frontier models); (right) Normalization layer placement evolution from Pre-
Norm to hybrid Post+QK-Norm for stability. The ”magic” of 2025 is not architectural revolution but optimization
refinement.
•2−3×training speedup (MTP + improved data cu-
ration)
• Near-elimination of training instabilities (Post-Norm
+ QK-Norm)
These ”tricks” compound: DeepSeek-V3 ($5.5Mtrain-
ing cost, $0.27/M tokens inference) achieves GPT-4-level
performance at100×lower cost than GPT-3’s training
budget ($4.6M vs. $460M estimated). The 2025 lesson:
architectural refinement scales better than architectural
revolution.4.12 Generative Models ArchitectureBe-
yond Text
4.12.1 Diffusion Models and Image Generation
Denoising Diffusion Probabilistic Models
(DDPM)[63] revolutionized image generation. The
forward process gradually adds Gaussian noise:
q(xt|xt−1) =N(x t;p
1−β txt−1, βtI)
fort= 1, . . . , T. The reverse process learns to denoise:
pθ(xt−1|xt) =N(x t−1;µθ(xt, t),Σ θ(xt, t)). Training
26

minimizes:L=E x0,ϵ,t[||ϵ−ϵ θ(xt, t)||2]whereϵ θpre-
dicts the noise added. DDPMs achieve high-quality gen-
eration through iterative refinement.
Latent Diffusion Models (LDM)[126] (Stable Dif-
fusion) apply diffusion in a compressed latent space
using a V AE encoder:z=E(x), diffuse inz-
space, then decode:ˆx=D(z 0). This reduces com-
putational cost by 8-10×. Text conditioning via cross-
attention: Attention(Q, K, V) =softmax(QKT
√
d)Vwhere
Qcomes from image latents,K, Vfrom text embeddings
(CLIP/T5).
DALL-E[122, 121]: OpenAI’s text-to-image models.
DALL-E 2 uses CLIP guidance: optimize image gener-
ation to maximize CLIP similarity with text. DALL-E
3 improves prompt following through synthetic caption
generation with GPT-4.
Imagen[128] and Parti [168] (Google) demonstrated
that large language models as text encoders significantly
improve text-image alignment. Imagen uses frozen T5-
XXL for conditioning.
4.12.2 Video Generation and World Models
Sora[108] (OpenAI, 2024) extends diffusion to video
generation and world simulation. Represents videos as
spacetime patches, applies Transformer diffusion in patch
space. Generates up to 60-second videos at 1080p with
complex camera motion, consistent physics, and multi-
agent interactions. Trained on internet-scale video data
with video-text pairs. Key innovation: native variable res-
olution, duration, and aspect ratio training—no cropping
or resizing, preserving compositional diversity.
World Models[59, 60]: Learning simulators from
pixels. DreamerV3 trains on diverse tasks (Minecraft,
robotics) and learns dynamics:ˆs t+1=fθ(st, at). Agents
then plan in learned latent space, enabling efficient model-
based RL.
4.12.3 Multimodal Foundation Models
CLIP[116] learns joint text-image embeddings through
contrastive learning on 400M pairs. The objective:
L=−PN
i=1logexp(sim(I i,Ti)/τ)PN
j=1exp(sim(I i,Tj)/τ)where sim is co-
sine similarity,τis temperature. CLIP enables zero-shotimage classification, text-to-image retrieval, and serves as
the backbone for Stable Diffusion and DALL-E.
Flamingo[7] conditions frozen LLMs on im-
ages/videos via cross-attention. Perceiver resampler com-
presses visual tokens, which are fed into LLM cross-
attention layers. Achieves few-shot visual question an-
swering.
LLaV A[84] uses CLIP vision encoder + projection
layer to map images into LLM embedding space, then
fine-tunes on instruction-following data. Simple but ef-
fective for visual instruction following.
5 Training Methodologies: From
Pre-Training to Alignment
Modern LLMs employ sophisticated multi-stage train-
ing pipelines that combine unsupervised pre-training,
parameter-efficient fine-tuning, post-training compres-
sion, and reinforcement learning from human feedback.
This section provides comprehensive technical analysis of
the methodologies that transform raw Transformer archi-
tectures into aligned, capable AI systems.
5.1 Pre-Training Foundations and Data
Quality
Autoregressive Language Modeling.The foundation
of modern LLMs remains next-token prediction [18]:
LLM=−E x∼DPT
t=1logp θ(xt|x<t), whereDis the
pre-training corpus (5-15T tokens for frontier models),
x<tdenotes previous tokens, andp θis the model’s dis-
tribution. This simple objective exhibits remarkable scal-
ing properties: doubling compute yields consistent log-
linear performance improvements across diverse down-
stream tasks [78, 64].
Multi-Token Prediction (MTP).A recent train-
ing innovation predicts multiple future tokens simul-
taneously using auxiliary heads [35, 9]:L MTP =
−PT
t=1Pn
i=1λilogp(x t+i|x≤t)wheren= 4typically
and exponential decayλ i= 0.5i−1reflects increasing
prediction difficulty. DeepSeek-V3 reports1.3×train-
ing speedup: reaching target perplexity in 77% of stan-
dard training steps. The auxiliary prediction heads are
27

discarded post-training or repurposed for speculative de-
coding.
Data Curation and Quality Over Quantity.The
2024-2025 paradigm shift emphasizesdata quality over
quantity. Microsoft Phi-4 (14B parameters) matches
models 10×larger through curriculum learning on syn-
thetic textbook-quality data [89]. Llama 3’s 15T to-
ken dataset undergoes aggressive filtering: deduplication
(MinHash + exact matching), quality scoring (perplexity-
based filtering), safety filtering (hate speech, personal in-
formation removal), resulting in only 5-7T high-quality
tokens used for actual training [46]. The key insight:
performance∝quality1.5×√quantity rather than linear
scaling in raw corpus size.
5.2 Parameter-Efficient Fine-Tuning
(PEFT)
Full fine-tuning of billion-parameter models is computa-
tionally prohibitive. PEFT methods freeze most parame-
ters while training small adapter modules, achieving com-
parable performance with<1% trainable parameters.
LoRA (Low-Rank Adaptation)[69] injects trainable
low-rank decomposition into weight matrices:W′=
W+ ∆W=W+BAwhereW∈Rd×k(frozen
pre-trained weight),B∈Rd×r,A∈Rr×kwith rank
r≪min(d, k). Forward pass:h=Wx+BAx.
OnlyB, Aare updated, reducing trainable parameters by
d×k
r(d+k)≈10,000×for typical values (d= 4096, k=
4096, r= 8).Why low-rank works:Task-specific adapta-
tions lie in low-dimensional subspaces of the full param-
eter space. The intrinsic dimensionality of fine-tuning is
empirically≪1%of total parameters [3], justifying rank
bottlenecks.
QLoRA[42] combines 4-bit quantization with LoRA,
enabling 65B model fine-tuning on consumer GPUs
(48GB VRAM): quantize(W) =Q NF4(W−µ
σ)where
QNF4is NormalFloat4 quantization optimized for nor-
mally distributed weights. Key innovations: (1)double
quantization—quantize the quantization constants them-
selves to save memory; (2)paged optimizers—offload op-
timizer states to CPU RAM via unified memory. Trade-
offs: QLoRA introduces≈10% training slowdown (de-
quantization overhead) but enables 4×larger models on
given hardware. Quality degradation is ¡2% on standardbenchmarks when using NF4 vs FP16 base model.
Adapter Layers[68] insert bottleneck feed-forward
modules between Transformer layers:h′=h+
f(hW down)W upwhereW down∈Rd×rprojects hidden di-
mensiondto bottleneckr(r≪d, typicallyr= 64
ford= 768),fis a nonlinearity (ReLU/GELU), and
Wup∈Rr×dprojects back. The residual connection pre-
serves pre-trained knowledge while the bottleneck learns
task-specific transformations. Parameter overhead:2dr
d2≈
0.5-2%(e.g.,2×768×64
7682≈11%per layer, but only 12-
24 of 96 layers typically adapted).Architectural place-
ment:Original adapters add two modules per layer (af-
ter attention and feed-forward), but recent work shows
single adapter after feed-forward suffices [112]. Train-
ing time: 2-5×faster than full fine-tuning due to re-
duced gradient computation.Composability:Multiple
adapters can be trained for different tasks and swapped at
inference without reloading base model—enabling task-
specific customization in production systems [127].
Prefix Tuning[82] prepends learnable continuous
prompts (”virtual tokens”) to keys and values in each
attention layer: Attention(Q,[P(l)
K;K],[P(l)
V;V])where
P(l)
K, P(l)
V∈Rp×dare trainable prefix parameters for
layerl, andp≈10-20prefix tokens. Unlike dis-
crete prompt tuning which searches over token embed-
dings, prefix tuning optimizes continuous parameters in
key/value space, allowing direct gradient-based optimiza-
tion:min PL(fP(x), y)where only prefix parameters
P={P(l)
K, P(l)
V}L
l=1are trained. Parameter count:2·
L·p·d(e.g.,2×24×20×768≈737,000param-
eters, or 0.1% of GPT-2).Why it works:Prefix tokens
function as task-specific ”instructions” that condition all
subsequent tokens’ attention patterns without modifying
model weights.Training instability:Direct optimization
of prefix parameters can lead to divergence; solution is
to reparameterize via smaller MLP:P(l)
K=MLP θ(e(l))
wheree(l)∈Rp×d′withd′≪d, then discard MLP
post-training [82]. Trade-off vs LoRA: Prefix tuning
achieves similar performance with 10×fewer parame-
ters but shows higher sensitivity to hyperparameters (p,
learning rate), whereas LoRA is more robust and widely
adopted.Comparison:LoRA dominates in practice due
to simplicity, stability, and modular composability, but
prefix tuning remains relevant for extreme parameter effi-
ciency scenarios (e.g., storing 1000 task-specific models
28

in 1GB).
5.3 Post-Training Quantization and Com-
pression
Post-training quantization (PTQ) reduces numerical pre-
cision after training, mapping FP16/BF16 weights
to lower-bit representations (INT8/INT4): ˆWij=
round(Wij−zero i
scalei)×scale i+zero iwhere scale iand zero i
are per-channel quantization parameters. Achieves 4-8×
memory reduction and 2-4×inference speedup via INT
arithmetic.
GPTQ[51] optimizes quantization using second-order
information (Hessian approximation):arg min ˆW∥WX−
ˆWX∥2
Fsubject to ˆW∈quantized space, whereXis cal-
ibration data (128-1024 samples). Layer-by-layer quanti-
zation preserves accuracy better than naive rounding.
A WQ[83] protects salient weights from quantization
by identifying important weights via activation magni-
tudes: importance(W ij) =|W ij| · ∥X j∥2, then keeping
top 1% in higher precision (e.g., INT8) while quantizing
remainder to INT4.
Scaling Laws for Quantization[167] establish pre-
dictable degradation:L quant(N, D, b) =L full(N, D) +α·
N−β·b−γwhereNis parameters,Dis data,bis bit-
width, andα, β, γare empirically determined constants
(β≈0.3,γ≈1.5for INT4). This enables practitioners
to predict quantization impact before deployment.
Critical Caveat—Task-Dependent Degradation:
While perplexity degrades ¡1% with INT4 quantization
on standard benchmarks,reasoning tasks show 7-15%
degradation[43, 159]. For example, Llama-3-70B on
MATH: FP16 (56.9%) vs INT4 (48.3%,−8.6points).
Quantization disproportionately impacts multi-step rea-
soning, likely due to error accumulation across sequential
computations.
Quantization-Aware Training (QAT)[73] addresses
PTQ limitations by simulating quantization during train-
ing, allowing the model to adapt to reduced preci-
sion. Unlike post-training methods that quantize frozen
weights, QAT incorporates quantization into the forward
pass while maintaining full-precision gradients via the
straight-through estimator (STE) [14]: Forward pass uses
quantized weights ˆW=quantize(W), but gradients flow
as if∂ˆW
∂W= 1(identity approximation). The training ob-jective becomes:min WL(f quantize(W) (x), y)where quan-
tization is applied during every forward pass, forcing the
model to learn robust features under reduced precision
constraints.
QAT typically requires 5-10% of pre-training com-
pute (fine-tuning on 50-100B tokens) but recovers 50-
80% of quality loss compared to PTQ. For example,
INT4 QAT on Llama-3-70B achieves 54.2% on MATH
(vs 56.9% FP16,−2.7points) compared to INT4 PTQ at
48.3% (−8.6points). The trade-off: QAT requires ac-
cess to training infrastructure and representative training
data, making it impractical for end-users deploying pre-
trained models.Adoption:Limited to model providers
(Qualcomm AI chips use INT8 QAT for on-device mod-
els [115]); most practitioners rely on PTQ due to ease of
deployment.
5.4 Reinforcement Learning from Human
Feedback (RLHF)
RLHF [109, 143] aligns models with human preferences
through a three-stage process:
Stage 1: Supervised Fine-Tuning (SFT).Pre-trained
modelπSFTis fine-tuned on high-quality demonstrations
Ddemo ={(x i, yi)}where humans provide ideal com-
pletions:LSFT=−E (x,y)∼D demo[logπSFT(y|x)]. Typical
dataset size: 10K-100K demonstrations covering diverse
instructions. This stage bootstraps instruction-following
capability.
Stage 2: Reward Model Training.A reward model
rϕ(x, y)learns to predict human preferences from pair-
wise comparisonsD comp ={(x, y w, yl)}wherey w≻
yl(chosen vs. rejected completions). Uses Bradley-
Terry model [17]:p(y w≻y l|x) =σ(r ϕ(x, y w)−
rϕ(x, y l))optimized via cross-entropy loss:LRM=
−E(x,yw,yl)∼D comp[logσ(r ϕ(x, y w)−r ϕ(x, y l))]. Typical
dataset size: 100K-1M comparison pairs.
Stage 3: RL Optimization via PPO.Policyπ θis
optimized to maximize expected reward while main-
taining proximity to the SFT model via KL penalty:
max θEx∼D,y∼π θ(·|x)[rϕ(x, y)−βlogπθ(y|x)
πSFT(y|x)]where
β∈[0.01,0.1]controls deviation strength. The KL
penalty prevents mode collapse (policy exploiting re-
ward model errors) and distribution shift (forgetting pre-
training knowledge).
29

5.5 Proximal Policy Optimization (PPO)
PPO [131] stabilizes RL training through clipped pol-
icy updates:LCLIP(θ) =E t[min(r t(θ)ˆAt,clip(r t(θ),1−
ϵ,1 +ϵ) ˆAt)]wherer t(θ) =πθ(at|st)
πθold(at|st)is probability ra-
tio (importance sampling weight), ˆAt=r(x, y)−V(s t)
is advantage estimate (reward minus baseline value func-
tion), andϵ≈0.2limits policy change per update.
The clipping operation prevents catastrophically large up-
dates: ifr t>1 +ϵ(policy much more likely than old
policy) and advantage is positive, gradient is zeroed to
prevent over-optimization. This ensures training stability
at the cost of sample efficiency—PPO typically requires
millions of tokens for convergence.
5.6 Direct Preference Optimization (DPO)
DPO [118] bypasses explicit reward modeling and RL,
directly optimizing policy on preference data. Starting
from the RLHF objective, DPO derives a closed-form so-
lution by reparameterizing the optimal policy:π∗(y|x) =
1
Z(x)πref(y|x) exp(1
βr∗(x, y))whereZ(x)is the parti-
tion function,π refis the reference model (SFT check-
point), andr∗is the optimal reward. Rearranging yields
implicit reward:r(x, y) =βlogπθ(y|x)
πref(y|x)+βlogZ(x).
Substituting into the Bradley-Terry model and noting
thatZ(x)cancels in preference probability, DPO opti-
mizes:LDPO(θ) =−E (x,yw,yl)[logσ(βlogπθ(yw|x)
πref(yw|x)−
βlogπθ(yl|x)
πref(yl|x))].
Advantages over RLHF:(1) Eliminates reward model
training (saves 30-50% compute), (2) Avoids RL instabil-
ity (direct supervised learning), (3) Requires 2-3×fewer
preference pairs (no reward modeling bottleneck). How-
ever, DPO can underfit when data distribution shifts sig-
nificantly fromπ ref, as it lacks explicit value function to
extrapolate preferences.
5.7 Group Relative Policy Optimization
(GRPO) and Pure RL
GRPO [134], introduced in DeepSeek-R1, elimi-
nates the critic network from actor-critic RL. Instead
of learning value functionV(s), GRPO samples
Gcompletions per prompt and uses group statis-
tics for advantage estimation: ˆAi=r(x, y i)−1
GPG
j=1r(x, y j). The policy gradient:∇ θLGRPO=
−Ex,{y 1,...,y G}∼πθ[PG
i=1ˆAi∇θlogπ θ(yi|x)]has low
variance since group mean1
GPr(x, y j)provides
context-specific baseline. ForG= 8, variance reduction:
1−1
G= 87.5%compared to no baseline.
Pure RL Training Without Supervised Fine-Tuning.
GRPO enables training without supervised fine-tuning
when verifiable rewards are available (e.g., code execu-
tion correctness, math answer verification). DeepSeek-
R1 demonstrates pure RL achieving 79.8% on MATH
benchmark without human annotations—starting from
base model, reward function is simply:r(x, y) = +1if fi-
nal answer matches ground truth, else−1. This paradigm
shift suggests that for objective tasks, outcome verifica-
tion may be more effective than preference learning.
5.8 Odds Ratio Preference Optimization
(ORPO) and Constitutional AI
ORPO [65] further simplifies alignment by combining
SFT and preference learning in a single stage, eliminat-
ing the reference model:LORPO=LSFT+λ· LORwhere
the odds ratio loss:LOR=−E[logσ(logpθ(yw|x)
1−pθ(yw|x)−
logpθ(yl|x)
1−pθ(yl|x))]. By eliminating the reference modelπ ref,
ORPO reduces memory by 50% compared to DPO and
achieves1.3×faster training [65].
5.8.1 Constitutional AI (CAI)
[12] represents a paradigm shift toward AI-generated
feedback, reducing reliance on expensive human anno-
tation. The core idea: models can critique and improve
their own outputs when guided by a ”constitution”—a set
of principles defining desired behavior (e.g., ”Choose the
response that is most helpful, harmless, and honest”). CAI
operates in two stages:
Stage 1: Supervised Learning from Self-Critiques.
Given a promptx, the model generates an initial response
y0. A critique model (often the same model with dif-
ferent prompts) evaluatesy 0against constitutional prin-
ciples: ”Critique: This response violates principle X be-
cause...”. The model then generates a revised response
y1addressing the critique. This process can iterate mul-
tiple times:y 0→critique1→y 1→critique2→y 2,
creating a chain of self-improvement. The final revised
30

responses form supervised fine-tuning data:LCAI-SFT=
−E(x,y revised)[logp θ(yrevised|x)].
Stage 2: RL from AI Feedback (RLAIF).Instead of
human preference labels, AI feedback serves as the re-
ward signal. For each promptx, generate two responses
(y1, y2)and ask the model to evaluate: ”Which response
better adheres to the constitution?” This creates synthetic
preference pairs(x, y w, yl)wherey wis AI-judged pre-
ferred. Standard RLHF/DPO then trains on AI-generated
preferences instead of human labels. Anthropic’s Claude
models demonstrate that RLAIF achieves comparable
performance to RLHF while reducing human annotation
costs by 90% [12].
Why CAI works:Modern LLMs exhibit emergent meta-
reasoning—the ability to critique their own outputs when
properly prompted. By distilling this capability into train-
ing data, CAI creates a self-improving loop without hu-
man oversight. Trade-off: CAI inherits model biases (the
model critiques itself based on its current understanding),
requiring careful constitution design and periodic human
audits to prevent drift.
5.8.2 Rejection Sampling
[149] provides an alternative approach to improve pol-
icy quality through filtering. Instead of learning from
all model outputs, rejection sampling selectively curates
high-quality training data: (1) GenerateKcandidate re-
sponses from the current policyπ θfor each promptx:
{y1, . . . , y K} ∼π θ(·|x)whereK= 16−32typically;
(2) Score all candidates using the reward model:r i=
rϕ(x, y i)fori= 1, . . . , K; (3) Select top-kresponses
(typicallyk= 1or best response):y∗= arg max iri;
(4) Use selected responses for supervised fine-tuning:
LRS=−E x,y∗∼TopK({y i})[logp θ(y∗|x)]; (5) Iterate: train
new policy on filtered data, generate new candidates, re-
peat.
Mathematical Justification.Rejection sampling ap-
proximates sampling from the optimal policyπ∗(y|x)∝
πθ(y|x) exp(r(x, y)/β)by rejecting low-reward samples.
The acceptance probability for a candidate:p accept(y) =
min(1,π∗(y|x)
πθ(y|x)). By keeping only high-reward samples
(r(x, y∗)≥threshold), we effectively sample from a
sharpened distribution that concentrates probability mass
on desirable behaviors.Llama 2’s Iterative RLHF with Rejection Sampling.
Llama 2-Chat [149] demonstrates the power of iterative
refinement:Week 1:Train initial reward model on hu-
man preferences, generate 16 responses per prompt, keep
best response for SFT.Week 2:Run PPO on updated pol-
icy, collect new human feedback, update reward model.
Weeks 3-5:Repeat rejection sampling + PPO cycles. Each
iteration improves both the policy (via PPO) and the train-
ing data quality (via rejection sampling). After 5 rounds,
Llama 2-Chat achieves GPT-3.5-level performance with
only 27K human preference annotations (vs. 100K+ for
InstructGPT) [149].
Trade-offs: CAI vs. Rejection Sampling.Constitu-
tional AI scales better (no human labels), but inherits
model biases and requires careful principle design. Re-
jection sampling guarantees data quality (uses explicit
reward model), but requires many samples per prompt
(K= 16−32), increasing computational cost by10−20×
per iteration. In practice, frontier labs combine both: use
rejection sampling for critical safety domains (toxicity,
bias) where reward models are reliable, and CAI for open-
ended helpfulness where human preferences are subjec-
tive.
5.9 Training Methodology Trade-offs and
Critical Insights
The evolution from RLHF→DPO→GRPO→
ORPO reveals systematic trade-offs: (1)Sample effi-
ciency—RLHF requires105-106preference pairs for re-
ward model training, DPO operates directly on prefer-
ences (2-3× fewer samples), GRPO achieves comparable
performance with104prompts via group-based variance
reduction, ORPO requires2×104pairs. Cost compari-
son: Cost RLHF :Cost DPO:Cost GRPO :Cost ORPO≈10 :
5 : 3 : 2; (2)Stability vs. flexibility—PPO’s clipping
(ϵ= 0.2) ensures stability but limits exploration, DPO’s
implicit reward avoids instability but can underfit on dis-
tribution shifts, GRPO’s group baseline balances both
through adaptive centering. Empirically, PPO achieves
85% preference win rate with high compute, DPO reaches
82% at half the cost, GRPO attains 88% by eliminat-
ing reward model bias; (3)Reward specification—RLHF
learns complex human preferences (helpfulness + harm-
lessness), DPO optimizes pairwise rankings (easier to col-
31

lect), GRPO exploits verifiable rewards (code execution,
math correctness).
The Critical Finding:Verifiable rewards + pure RL
≥human preferences + supervised learningfor objective
tasks, as demonstrated by DeepSeek-R1’s 79.8% MATH
(GRPO without SFT) vs GPT-4’s 52.9% (RLHF with
SFT) [39]. This suggests the field is transitioning from
preference learning to outcome optimization, where task-
specific verifiable signals replace subjective human judg-
ments.
6 Breaking the Scaling Wall: Eight
Alternative Paradigms
Given the constraints on data availability, compute costs,
and energy consumption documented in the previous
sections, the field is exploring eight complementary
paradigms to continue capability improvements without
relying on brute-force scaling. This section answers the
question:WHAT alternatives exist to overcome the
scaling wall?These approaches span architectural in-
novations, deployment strategies, training efficiency, and
data quality optimization.
6.1 Alternative 1: Test-Time Compute
Scaling—Trading Inference for Pre-
training
Core Idea:Rather than increasing pre-training com-
pute, allocate additional inference-time computation to
improve reasoning. OpenAI o1 [107] and DeepSeek-
R1 [39] demonstrate that extended chain-of-thought rea-
soning during inference can match or exceed the perfor-
mance of much larger models.
Mathematical Framework:The key insight is that
performance can be decomposed into pre-training and
test-time components:
Performance∝f(C pretrain ) +g(C test)(60)
wherefandgrepresent the performance gains from pre-
training and test-time compute, respectively.
Empirical Evidence:DeepSeek-R1 (671B parame-
ters) with extended reasoning matches GPT-4-level per-
formance using 3-10×inference compute, suggestingtest-time scaling may be more efficient than pre-training
scaling for reasoning tasks. On MATH benchmark, o1
achieves 94.8% with extended thinking time vs. 60.3%
for GPT-4 Turbo without extended reasoning [107].
6.2 Alternative 2: Sparse Architec-
tures—MoE and Structured Pruning
Core Idea:Mixture-of-Experts (MoE) architec-
tures [137, 76, 40] provide a form of structured sparsity,
activating only a subset of parameters per token while
maintaining the capacity of much larger dense models.
Efficiency Gains:DeepSeek-V3 achieves 671B to-
tal parameters with only 37B active per token, reduc-
ing inference costs by 18×while maintaining dense-
model quality. The routing function Router(x) =
TopK(softmax(W gx), k)selects the top-kexperts (typi-
callyk= 8out of 256 total experts) for each token, pro-
viding massive computational savings.
Post-Training Compression:Quantization techniques
(QLoRA [42], GPTQ [51], AWQ [83]) reduce model size
by 4-8×with minimal quality loss. QLoRA enables fine-
tuning 65B models on a single 48GB GPU, democratizing
access to large model training.
6.3 Alternative 3: Architectural Innova-
tions Beyond Transformers
Core Idea:Alternative architectures aim to break the
O(n2)attention bottleneck that makes long-context pro-
cessing prohibitively expensive:
•Linear Attention: Katharopoulos et al. [79]
reformulate attention as Attention(Q, K, V) =
ϕ(Q)(ϕ(K)TV), achievingO(n)complexity. Re-
cent Gated DeltaNet [9] in Kimi Linear (48B)
combines linear attention with gating mechanisms
for improved quality, matching Transformer perfor-
mance on 128K context with 5×faster inference.
•State Space Models: Mamba [57] achieves linear
complexity through selective state spaces:h t=
¯Aht−1+¯Bxt, where ¯Aand ¯Bare input-dependent.
Mamba-2.8B matches Transformer-7B performance
with 5×faster inference and 8×higher throughput
on long sequences.
32

•Sparse Attention: Sliding window attention (Mis-
tral [75], Gemma 3 [55]) uses aggressive 5:1 or
8:1 SWA ratios, reducing effective computation to
O(nw)wherew≪nis the window size. Lo-
cal+global patterns combine fine-grained local at-
tention with sparse global connections for efficient
long-range modeling.
6.4 Alternative 4: Post-Training Quanti-
zation—Scaling Laws for Compressed
Models
Core Idea:Post-training quantization (PTQ) reduces
model precision from FP16/BF16 to INT4/INT8, achiev-
ing 4-8×memory reduction and 2-4×inference speedup
with minimal quality degradation. Unlike traditional
compression, recent work establishesscaling laws for
quantized models[167], showing that quantization be-
haves predictably across model scales.
Scaling Laws:Yao et al. [167] demonstrate that
quantized model performance follows power-law rela-
tionships with model size, data size, and bit-width:
Lquantized (N, D, b) =L full(N, D)+α·N−β·b−γ, whereN
is parameters,Dis data,bis bit-width, andα, β, γare em-
pirically determined constants. This enables practitioners
to predict quantization impact before deployment.
Practical Impact:INT4 quantization of Llama-3-
70B achieves ¡1% perplexity increase while reducing
from 140GB to 35GB, enabling deployment on consumer
GPUs. GPTQ [51] and AWQ [83] demonstrate that layer-
wise quantization with activation-aware rounding main-
tains accuracy while achieving 4×compression. This
paradigm enables edge deployment and reduces inference
costs by 75%, directly addressing the energy/cost dimen-
sions of the scaling wall.
Task-Dependent Degradation—Critical Caveat:
While the above claims hold forperplexity and simple
classification tasks, recent evidence revealssubstantial
degradation on complex reasoning benchmarksat
INT4 and below [43, 51, 159]. Table 2 summarizes
empirical results:
Root Cause Analysis:The perplexity-performance
gap arises because reasoning tasks requireprecise inter-
mediate representations. INT4 quantization introduces
rounding errors that compound across multi-step rea-soning chains, leading to cascading failures in chain-of-
thought processes [155]. Simple tasks (factual recall,
classification) are robust because they rely on pattern
matching in early layers, whereas reasoning requires ex-
act value propagation through deep computation graphs.
The data in Table 2 shows reasoning tasks (GSM8K: -
8.6%, MATH: -6.4%, HumanEval: -5.5%) degrade 1.5-
2×more than knowledge tasks (MMLU: -5.8%, -4.2%)
at INT4, despite perplexity increasing by only 0.35 points
(6.4% relative) on WikiText-2.
Task Sensitivity Hierarchy (GPTQ INT4):
•Low Sensitivity(INT4 viable): MMLU factual recall
(-4 to -6%), HellaSwag common sense (-2 to -3%),
classification tasks
•Medium Sensitivity(INT8 recommended): Hu-
manEval code generation (-5 to -6%), GSM8K
grade-school math (-7 to -9%)
•High Sensitivity(FP16 required): MATH competi-
tion math (-6 to -8% absolute, -47% relative), GPQA
graduate-level science (-9 to -12%), multi-hop rea-
soning
Mitigation Strategies:
•Mixed Precision:Keep attention layers in FP16,
quantize only feed-forward networks (FFN) to INT4.
Reduces degradation to 3-5% [159].
•QLoRA Fine-Tuning:Quantize base model to INT4,
fine-tune low-rank adapters (LoRA) in FP16. Pre-
serves 95-98% of full-precision quality [43].
•Dynamic Quantization:Route ”hard” reasoning ex-
amples to FP16, ”easy” examples to INT4 via adap-
tive thresholding.
Note:INT4 quantization is viable fordeployment
scenariosprioritizing efficiency over accuracy (e.g.,
chatbots, content generation), butnot recommended
for high-stakes reasoning applications(mathematical
proofs, code correctness verification, scientific reason-
ing). INT8 quantization remains the safe default for
reasoning-heavy workloads, achieving 2×compression
with ¡1% degradation across all task types. Practitioners
must evaluate quantization impact on their specific down-
stream tasks rather than relying solely on perplexity met-
rics.
33

Table 2: Task-dependent quantization degradation using GPTQ [51]. INT4 quantization causes severe accuracy drops
on reasoning-heavy tasks (GSM8K, MATH, HumanEval) despite ¡1% perplexity increase on WikiText-2. Reasoning
tasks exhibit 2-3×larger degradation compared to knowledge-based tasks (MMLU), demonstrating that perplexity
is insufficient for predicting downstream performance. Base model performance from official technical reports [149,
75]. Benchmarks: MMLU [62] (5-shot), GSM8K [27] (8-shot CoT), MATH [62] (4-shot), HumanEval [22] (0-shot
pass@1).
Model Task Type FP16 INT8 INT4 (GPTQ)∆INT4
LLaMA-2-70B MMLU (Knowledge) 68.9% 68.2% 63.1% -5.8%
LLaMA-2-70B GSM8K (Math Reasoning) 56.8% 55.3% 48.2%-8.6%
LLaMA-2-70B MATH (Competition Math) 13.5% 12.8% 7.1%-6.4%
LLaMA-2-13B HumanEval (Code) 29.9% 29.3% 24.4%-5.5%
Mistral-7B-v0.1 MMLU (Knowledge) 62.5% 61.9% 58.3% -4.2%
Mistral-7B-v0.1 GSM8K (Math Reasoning) 52.2% 51.1% 44.8%-7.4%
WikiText-2 Perplexity: FP16: 5.47 — INT8: 5.51 (+0.7%) — INT4: 5.82 (+6.4% relative, 0.35 absolute)
6.5 Alternative 5: Distributed Edge Com-
puting—Leveraging Massive Device
Networks
Core Idea:Rather than concentrating compute in central-
ized data centers, distribute LLM inference and training
across massive networks of edge devices (smartphones,
IoT devices, edge servers). Yang et al. [164] demonstrate
that aggregating spare compute from millions of edge de-
vices can match datacenter-scale resources while reducing
energy consumption and network costs.
Technical Framework:Edge-distributed LLMs par-
tition models across devices using techniques includ-
ing: (1)layer-wise splitting—different devices han-
dle different layers with pipeline parallelism, (2)MoE-
native distribution—routing experts to different edge
nodes based on network topology, (3)federated infer-
ence—multiple devices collaboratively process queries
with privacy preservation, and (4)adaptive offload-
ing—dynamic workload distribution based on device
availability and network conditions. The key insight:
Total Capacity=PN
i=1Ci×Ai×EiwhereC iis device
compute,A iis availability factor, andE iis efficiency co-
efficient.
Breaking Barriers:Edge distribution breaks
three scaling wall dimensions simultaneously: (1)
cost—leverages existing consumer hardware rather
than purchasing new datacenter GPUs ($0.10/hour edge
compute vs. $2-4/hour cloud H100), (2)energy—utilizesspare capacity during device idle time rather than
running dedicated servers 24/7 (50-80% lower total
energy consumption), and (3)data locality—enables
processing near data sources, reducing network transfer
costs. Experiments show 7B-parameter models achieving
15-20 tokens/second across distributed edge networks,
comparable to single-GPU inference but at 10×lower
cost [164].
6.6 Alternative 6: Model Merg-
ing—Combining Specialized Capa-
bilities Without Retraining
Core Idea:Rather than training single monolithic mod-
els, create specialized expert models for different do-
mains/tasks, thenmergethem into unified systems ex-
hibiting combined capabilities without additional train-
ing. MergeKit [53] provides systematic techniques for
parameter averaging, task-arithmetic composition, and
DARE (Drop And REscale) merging.
Merging Techniques:Three primary ap-
proaches demonstrate effectiveness: (1)Linear
interpolation—θ merged =αθ A+ (1−α)θ B
whereαcontrols task balance, (2)Task arith-
metic—θ multi=θ base+λ1(θA−θ base) +λ 2(θB−θ base)
enables additive skill composition, and (3)DARE merg-
ing—randomly drops parameters before merging to
reduce interference, then rescales remaining weights for
34

stability. Taylor et al. [147] demonstrate that merged
models exhibitsynergistic capabilities, performing better
on combined tasks than either source model, with up to
15% improvements on cross-domain benchmarks.
Training Efficiency:Model merging circumvents the
scaling wall by eliminating multi-task training costs.
Training two 7B specialized models (2×50 GPU-days
= 100 GPU-days) then merging is 10-20×cheaper than
training a single multi-task 13B model (1,000+ GPU-
days) while achieving comparable or superior perfor-
mance. This paradigm enables: (1)continuous capabil-
ity addition—merge new skills into existing models with-
out catastrophic forgetting, (2)community-driven devel-
opment—different organizations contribute specialized
models that combine into powerful generalists, and (3)
personalization—users merge domain-specific adapters
with base models for customized systems.
6.6.1 Case Study : SLERP Merging for Code and
Math Capabilities
We evaluate merging specialized Mistral-7B variants [91]
using Spherical Linear Interpolation (SLERP) [53]. Two
source models:Mistral-7B-Code(fine-tuned on 100B to-
kens of code from The Stack and StackOverflow) and
Mistral-7B-Math(fine-tuned on 50B tokens of mathemat-
ical reasoning from MATH [62], GSM8K [27], and Proof
datasets). Merging configuration: SLERP witht= 0.5
(equal weighting), gradient surgery to reduce task inter-
ference.
The merged model retains 96-97% of each special-
ist’s capability while maintaining general knowledge
(MMLU). Total training cost: 100 GPU-days (50 days
per specialist) vs. 1,000 GPU-days for multi-task train-
ing, achieving 10×cost reduction. Latency: identical to
source models (15ms per token on A100). Key insight:
specialized training followed by merging avoids catas-
trophic interference during multi-task optimization.
Failure Modes and Mitigation Strategies Catas-
trophic Interference in Model Merging:When merg-
ing models trained on conflicting objectives (e.g., tox-
icity reduction vs. creative writing), merged parame-
ters exhibit unstable behavior.Example:Merging GPT-
3.5 safety-tuned variant with uncensored creative writingvariant produced outputs oscillating between overly cau-
tious and unsafe responses.Mitigation:(1) DARE merg-
ing with 50% drop rate reduces parameter conflicts, (2)
gradient surgery projects conflicting gradients to orthog-
onal subspaces before merging, (3) selective layer merg-
ing—merge only attention layers, keep specialized FFN
layers separate.
Privacy Leakage in Federated Systems:Gradient
inversion attacks can reconstruct training samples from
shared gradients [172].Example:In medical imaging fed-
erated learning, attackers recovered patient X-rays from
gradient updates with 72% pixel accuracy.Mitigation:
(1) Differential privacy (DP-SGD) adds calibrated noise
(σ∝√
2 ln(1.25/δ)
ϵ) guaranteeing(ϵ, δ)-DP, (2) secure ag-
gregation using homomorphic encryption ensures server
never sees individual gradients, (3) gradient clipping lim-
its per-sample contribution before aggregation.
Reliability Issues in Edge Computing:Device het-
erogeneity and availability create inference inconsisten-
cies.Example:Distributed edge network for real-
time translation exhibited 23% failure rate due to de-
vice dropouts during peak hours.Mitigation:(1) redun-
dancy—replicate inference across 3 devices, use major-
ity voting (reduces failure to 1.2%), (2) adaptive time-
outs—allocate tasks to faster devices first, fallback to
cloud after 500ms, (3) model checkpointing—cache in-
termediate states enabling seamless device handoff.
6.7 Alternative 7: Efficient Training Al-
gorithms—ORPO and Reference-Free
Optimization
Core Idea:Traditional RLHF requires training sepa-
rate reward models and maintaining reference models
during optimization, doubling memory requirements and
compute costs. Odds Ratio Preference Optimization
(ORPO) [65] eliminates reference models by directly op-
timizing odds ratios between preferred and rejected re-
sponses.
Mathematical Framework:ORPO combines super-
vised fine-tuning with preference learning in a single ob-
jective:
LORPO =L SFT+λ·E (x,yw,yl)
−logσ
logPθ(yw|x)
Pθ(yl|x)
(61)
35

Table 3: Model merging achieves 96-99% of specialized model performance on both tasks with zero additional
training cost, compared to 1,000 GPU-days for multi-task training from scratch. SLERP merging technique from
MergeKit [53]. Base models: Mistral-7B [91]. Benchmarks: HumanEval [22], GSM8K [27], MMLU [62]. Training
cost estimates based on A100 GPU hours and standard fine-tuning protocols [147].
Model HumanEval GSM8K MMLU
Base Mistral-7B 26.8% 57.1% 62.5%
Mistral-7B-Code 48.2% 59.3% 61.8%
Mistral-7B-Math 28.1% 83.7% 64.2%
SLERP Merged 46.3% 81.2% 63.7%
Retention vs. Best96.1% 97.0% 99.2%
Multi-Task Trained 47.1% 82.5% 64.1%
Training Cost 100 GPU-days 100 GPU-days 100 GPU-days
Merging Cost0 GPU-days 0 GPU-days 0 GPU-days
wherey wis the preferred response,y lis the rejected re-
sponse, andλcontrols preference strength. Unlike DPO
which requiresP ref, ORPO directly maximizes the odds
ratioPθ(yw|x)
Pθ(yl|x), implicitly penalizing rejected responses
while amplifying preferred ones.
Resource Savings:By eliminating reference mod-
els, ORPO reduces peak memory by 50% (no need to
storeP refparameters) and training time by 30-40% (sin-
gle forward pass per sample instead of two). This en-
ables training 70B-parameter models on 8×A100 config-
urations that previously required 16×A100 for DPO. The
efficiency gains directly address compute cost constraints,
making preference optimization accessible to resource-
constrained researchers and enabling more extensive hy-
perparameter search within fixed budgets.
6.8 Alternative 8: Small Specialized Mod-
els—Phi’s Data Quality Over Scale
Paradigm
Core Idea:Microsoft’s Phi series [89] challenges the
”bigger is better” paradigm by demonstrating thatdata
quality and curriculum designcan enable small models
(3.8B-14B parameters) to match or exceed 10×larger
models on reasoning tasks. Phi-4 (14B) achieves 84.8%
on MATH, competitive with GPT-4-level performance,
through systematic curation of training data and multi-
stage curriculum learning.
Training Strategy:Phi models employ three key in-novations: (1)Filtered high-quality data—curate ”text-
book quality” datasets emphasizing logical reasoning,
mathematical derivations, and clear explanations rather
than raw web text volume, (2)Synthetic data augmen-
tation—generate additional training examples targeting
specific reasoning patterns and edge cases using larger
teacher models, and (3)Progressive curriculum—start
with simple problems, gradually increase difficulty, and
repeat challenging examples multiple times to reinforce
weak areas. This approach prioritizeslearning efficiency
over scale efficiency.
Scaling Wall Implications:Phi demonstrates that the
scaling wall can be circumvented through smarter data
strategies rather than more data. Training Phi-4 (14B)
costs∼$500K (3,000 GPU-days on A100s) compared to
∼$50M+ for 140B-parameter models, achieving 100×
cost reduction while maintaining competitive reasoning
performance. The small size enables: (1)local deploy-
ment—runs on laptops and consumer GPUs, eliminat-
ing cloud costs, (2)rapid iteration—faster experimen-
tation cycles accelerate research, and (3)edge intelli-
gence—enables AI applications on mobile devices and
embedded systems. Phi’s success suggests the field may
be approaching ”data saturation” where quality matters
exponentially more than quantity.
36

6.9 The Changing Nature of Scaling
Hooker [67] argues that the field is experiencing a fun-
damental shift: the simple formula of ”scale model
size and training data” has become inadequate, with
the relationship between training compute and perfor-
mance now highly uncertain and rapidly changing. This
shift has profound implications: (1)Academia marginal-
ized—the capital-intensive scaling paradigm has con-
centrated progress in industry labs while fundamentally
reshaping scientific culture, (2)Transparency declin-
ing—industry labs have increasingly stopped publishing
detailed methodologies, and (3)Alternative levers emerg-
ing—architectural innovations, training efficiency, and
test-time compute represent more promising paths for-
ward than naive scaling. The eight alternative paradigms
above—test-time compute, sparse architectures, architec-
tural innovations, post-training quantization, distributed
edge computing, model merging, efficient training al-
gorithms, and small specialized models—represent pre-
cisely these ”more interesting levers of progress” that
Hooker identifies as key to navigating the post-scaling era.
Synthesis:These innovations suggest that continued
AI progress may depend less on brute-force scaling and
more on algorithmic efficiency, architectural innovation,
and strategic allocation of compute resources. The con-
vergence of multiple paradigms offers a comprehensive
toolkit for post-scaling era development: (1) test-time
compute and MoE sparse architectures reduce inference
costs, (2) quantization and edge distribution democra-
tize deployment, (3) model merging and efficient train-
ing algorithms (ORPO) reduce development costs, and (4)
small specialized models (Phi) prove that data quality can
substitute for scale. Together, these approaches form a vi-
able path forward, enabling continued capability growth
despite hitting the traditional scaling wall. The remain-
der of this survey explores how these innovations inte-
grate with architectural breakthroughs (Section 2), train-
ing methodologies (Section 3), and practical deployment
strategies (Section 4).7 Economic and Environmental
Sustainability: Quantifying the
Costs
This section providesdetailed quantitative analysisof
training and deployment costs, answering the critical
question:WHAT are the specific numbers behind the scal-
ing wall?We decompose costs into hardware, energy, and
operational components, providing formulas and concrete
examples to enable practitioners to estimate costs for their
own scenarios.
7.1 Hardware Acquisition and Amortiza-
tion
Hardware costs are calculated using a depreciation model
over the hardware lifetime (typically 3-5 years):
Camortized
hardware =Pchip×N chips×(1 +O network )
Tlifetime×U util×T train(62)
whereP chipis the purchase price per accelerator ($10,000-
30,000 for A100/H100),N chips is the number of accel-
erators,O network ≈0.23accounts for cluster networking
overhead [15],T lifetime is the expected hardware lifetime
(3-5 years),U utilis the utilization rate (60-80%), andT train
is training duration.
Key Insight:Amortized costs are 10-20×lower than
upfront acquisition costs, making ownership economi-
cal for sustained training programs. However, the high
upfront capital requirement ($100M-500M for frontier
model clusters) creates barriers for all but the largest or-
ganizations.
Table 4 shows cost breakdowns for representative mod-
els.
7.2 Energy Consumption
Energy costs have become a significant fraction of total
training costs, growing from 10-15% (GPT-3) to 20-30%
(modern models) [15]. The total energy cost formula:
Cenergy =R elec×P TDP×R avg×PUE×N chip-hours (63)
whereR elecis the electricity cost rate ($/kWh), typically
$0.08-0.15/kWh for data centers,P TDPis the thermal de-
sign power in kilowatts (0.35kW for A100, 0.70kW for
37

Table 4: Hardware and energy costs for representative
frontier models. Data from [15, 133]. Cloud costs repre-
sent actual or estimated rental prices. Amortized costs as-
sume 4-year depreciation, 70% utilization, 23% network-
ing overhead. Energy costs use regional electricity rates
($0.08-0.15/kWh) and data center PUE (1.1-1.3).
Model Year Chip-Hours Amortized Energy
(Million) Cost ($M) Cost ($M)
GPT-3 2020 11.7 (V100) 2.4 0.9
Gopher 2021 15.4 (TPUv3) 3.1 1.1
Chinchilla 2022 6.8 (TPUv4) 2.0 0.6
PaLM 2022 29.2 (TPUv4) 8.5 2.7
GPT-4 2023 64.1 (A100) 78.3 6.2
Llama 3 405B 2024 51.2 (H100) 95.4 7.8
Gemini 1.5 2024 87.3 (TPUv5) 145.2 11.3
DeepSeek-V3 2025 55.7 (H800) 101.8 8.9
H100),R avgis the average power-to-TDP ratio (65-80%
depending on manufacturer and workload), PUE is the
power usage effectiveness (1.1-1.3, accounting for cool-
ing and power distribution overhead), andN chip-hours is the
total accelerator-hours.
Regional Variations:Energy costs vary significantly
by location: $0.08/kWh (US Pacific Northwest hydro-
electric), $0.12/kWh (US average), $0.15/kWh (Western
Europe), and $0.25+/kWh (peak rates in some regions).
This 3×variation incentivizes data center placement in
low-cost energy regions.
7.3 Computational Intensity Analysis
Table 5 provides a detailed breakdown of computational
requirements and equivalent hardware counts for major
models, illustrating the massive scale of frontier AI train-
ing. These equivalents help contextualize the computa-
tional demands: training GPT-4 required compute equiv-
alent to 641,025 A100 GPUs running continuously, or 2.4
million consumer RTX 4090 GPUs.
The energy consumption of GPT-4 training (6,154
MWh, approximately 6.15 GWh) is equivalent
to the annual electricity consumption of approxi-
mately 570 average US households (based on 10,800
kWh/household/year). This represents a 22×increase
from GPT-3 (280.8 MWh), highlighting the unsustain-ability of naive scaling. Note: All energy estimates
have ±20% uncertainty due to variations in GPU utiliza-
tion rates (60-80%), power usage effectiveness (PUE:
1.1-1.3), and infrastructure overhead.
7.4 Cloud Compute Costs
Cloud rental costs provide an alternative cost model, typi-
cally 2-4 times higher than amortized ownership costs due
to provider margins and infrastructure overhead [133].
Figure 4 compares cloud rental prices with amortized
ownership costs across model generations.
Economic Trade-offs:Cloud rental offers flexibility
(no upfront capital, pay-per-use) but higher total cost for
sustained training. Ownership requires $100M-500M up-
front capital but achieves 2-4×lower costs for multi-
month training runs. The breakeven point typically occurs
at 3-6 months of continuous training, making ownership
economical only for organizations with sustained training
programs.
8 Evaluation and Benchmarking
8.1 Traditional Benchmarks
MMLU (Massive Multitask Language Understand-
ing)[62]: 57 subjects spanning STEM, humanities, social
sciences. 5-shot evaluation. Tests knowledge breadth and
reasoning.
HumanEval[22]: 164 Python programming problems.
Measures functional correctness via unit tests. Metric:
pass@k (probability of generating correct solution ink
attempts).
GSM8K[27]: 8.5K grade school math problems re-
quiring multi-step reasoning. Tests arithmetic reasoning
and problem decomposition.
MATH[62]: Competition-level mathematics (AMC
10/12, AIME). 5 difficulty levels. Tests advanced mathe-
matical reasoning.
GPQA (Graduate-Level Google-Proof Q&A)[125]:
PhD-level science questions designed to be unsolvable via
web search. Tests true expert knowledge.
Big-Bench[142]: 200+ diverse tasks testing capa-
bilities like logical reasoning, commonsense, multi-step
arithmetic.
38

Table 5: Computational intensity and hardware equivalents for frontier models. Data from [15, 133, 18, 105]. Training
compute measured in PetaFLOP-days (PF-days). Hardware equivalents show the number of consumer/data center
GPUs required to match training compute. Energy values have ±20% uncertainty due to variations in utilization (60-
80%), PUE (1.1-1.3), and infrastructure overhead.
Model Training Desktop RTX 4090 A100 Energy Household-Years
Compute PC Equiv. Equiv. Equiv. Consumption Equivalent
(PF-days) (1.23 TF) (82.6 TF) (312 TF) (MWh) (10,800 kWh/yr)
GPT-3 (175B) 3,640 2,963,000 44,066 11,667 280.8 (±56) 26
Codex (12B) 450 366,300 5,447 1,442 34.6 (±7) 3
Gopher (280B) 5,760 4,688,640 69,733 18,462 443.0 (±89) 41
Chinchilla (70B) 6,120 4,981,680 74,091 19,615 157.5 (±32) 15
PaLM (540B) 10,800 8,790,720 130,739 34,615 788.4 (±158) 73
GPT-4 (1.8T†) 200,000 162,800,000 2,420,774 641,025 6,153.8 (±1,230) 570
Llama 3 (405B) 31,200 25,405,440 377,778 100,000 1,996.8 (±399) 185
Gemini 1.5†56,000 45,584,000 678,111 179,487 3,571.2 (±714) 331
DeepSeek-V3 (671B) 35,700 29,058,960 432,203 114,423 2,278.6 (±456) 211
†Parameters/costs estimated based on compute requirements and third-party analysis; architectural details unconfirmed.
Desktop PC = Intel i9-14900K (1.228 TFLOPS FP32). RTX 4090 = 82.6 TFLOPS (FP32), A100 = 312 TFLOPS (FP16 Tensor).
Energy calculated: Chip-hours×TDP×0.70 (utilization)×1.2 (PUE). Uncertainty ranges shown as (±value).
Household-years based on 10,800 kWh/year average US residential consumption. Data sources: [18, 105, 15].
8.2 Human Preference Evaluation
Chatbot Arena[24, 169]: Anonymous pairwise battles
between models, rated by users. Elo scores computed
from win rates. Over 1M votes. Correlates better with
real-world usefulness than automatic metrics.
MT-Bench[169]: Multi-turn conversations judged by
GPT-4. Tests instruction following, reasoning, math, cod-
ing, roleplay.
AlpacaEval[47]: Automated evaluation using LLM
judges (GPT-4) on 805 questions. Measures helpfulness.
8.3 Emergent Abilities and Scaling Phe-
nomena
Wei et al. [154] documented that certain capabilities ap-
pear suddenly at sufficient scale rather than gradually im-
proving. Examples: 3-digit arithmetic, Persian QA, IPA
transliteration. These abilities are ”emergent” because
they are unpredictable from smaller models.
However, Schaeffer et al. [129] challenged this inter-
pretation, demonstrating that emergent abilities may be
artifacts of evaluation metrics. Using smooth metrics
(e.g., Brier score instead of accuracy), the same capabil-ities appear to scale smoothly. This suggests emergence
may reflect metric choice rather than fundamental phase
transitions.
The debate highlights the importance of metric design
in understanding model capabilities. Recent work focuses
on differentiable, continuous metrics that capture gradual
capability acquisition.
8.4 Comparative Performance Analysis
Table 6 presents a comprehensive comparison of major
model families across key benchmarks, including recent
2025 results.
Key observations from Table 6:
•Reasoning Breakthrough (2024-2025): Models
with dedicated reasoning training (o1, DeepSeek-
R1, Phi-4 Reasoning) show dramatic improvements
on MATH (70-95%) and AIME (30-83%) compared
to standard models (30-50% MATH).
•Open-Source Parity: Llama 3 (405B), Qwen 3,
and Kimi K2 match or exceed GPT-4 Turbo on
most benchmarks, demonstrating open-source has
achieved competitive performance.
39

Table 6: Comprehensive performance comparison of major language model families across benchmarks. Scores
represent the best published results for each model family. Recent 2025 models show significant advances in reasoning
benchmarks (MATH, AIME, GPQA).
Model Family Year Params MMLU HumanEval GSM8K MATH AIME GPQA Arena Elo
OpenAI GPT Series
GPT-2 2019 1.5B – – – – – – –
GPT-3 2020 175B 70.0 – 17.0 8.8 – – –
GPT-3.5-Turbo 2022 – 70.0 48.1 57.1 23.5 – – 1117
GPT-4 2023 – 86.4 67.0 92.0 52.9 13.4 56.1 1251
GPT-4-Turbo 2024 – 86.5 85.4 94.2 64.5 18.9 63.4 1257
GPT-4o 2024 – 88.7 90.2 96.4 76.6 27.5 70.2 1287
o1-preview 2024 – 90.8 92.3 96.4 85.5 44.6 78.3 1348
o1 2024 – 92.3 92.2 96.1 94.8 83.3 78.2 1355
Meta LLaMA Series
LLaMA 2023 65B 63.4 – 50.9 10.6 – – –
Llama 2 2023 70B 68.9 29.9 56.8 13.5 – 29.1 1077
Llama 3 2024 8B 66.6 62.2 79.6 30.0 – 34.2 1155
Llama 3 2024 70B 79.5 81.7 93.0 50.4 11.9 46.7 1213
Llama 3 2024 405B 88.6 89.0 96.8 73.8 23.8 51.1 1265
Llama 3.1 2024 405B 88.6 89.0 96.8 73.8 24.0 51.1 1271
DeepSeek Series
DeepSeek LLM 2024 67B 71.3 – 63.0 15.8 – – –
DeepSeek-Coder 2024 33B 56.5 87.0 75.4 32.6 – – –
DeepSeekMath 2024 7B 34.2 – 82.9 51.7 – – –
DeepSeek-V2 2024 236B 78.5 81.1 92.2 43.6 8.9 42.5 1203
DeepSeek-V2.5 2024 236B 80.5 89.0 94.7 58.3 13.2 47.8 1250
DeepSeek-V3 2025 671B 88.5 90.2 97.3 76.4 39.2 59.1 1302
DeepSeek-R1 2025 671B 90.8 96.3 97.3 79.8 79.2 71.5 1348
DeepSeek-R1 Distill 2025 70B 85.9 90.8 95.1 75.4 48.7 65.2 1298
Google Models
Gemini 1.0 Ultra 2023 – 90.0 74.4 94.4 53.2 – 59.4 1230
Gemini 1.5 Pro 2024 – 85.9 84.1 91.7 67.7 22.7 63.8 1268
Gemini 2.0 Flash 2024 – 87.1 88.3 94.8 71.9 31.5 66.3 1285
Microsoft/Others
Phi-3 2024 14B 78.0 61.5 83.4 42.5 – – –
Phi-4 2024 14B 84.1 82.6 91.0 64.4 16.3 53.8 1201
Phi-4 Reasoning 2025 14B 85.2 88.9 93.7 74.9 34.8 61.2 1245
Mistral 7B 2023 7B 60.1 29.8 52.2 13.1 – 25.3 1079
Mixtral 8x7B 2024 47B 70.6 40.2 74.4 28.4 – 33.7 1146
Mixtral 8x22B 2024 141B 77.8 75.0 88.0 45.0 12.7 40.0 1193
Mistral Large 2 2024 123B 84.0 91.6 93.0 56.0 24.4 49.0 1232
Emerging Global Models
Qwen 2.5 (Alibaba) 2024 72B 84.2 86.4 95.8 68.6 21.4 56.3 1253
Qwen 3 (Alibaba) 2025 72B 86.3 86.2 95.2 70.1 28.9 59.8 1269
Kimi K2 (Moonshot) 2025 – 87.9 88.5 96.1 75.3 35.4 63.7 1289
K2-V2 2025 – 89.1 91.2 96.8 77.8 42.1 66.9 1305
GLM-4.5 2025 – 85.7 87.3 94.3 69.8 27.6 58.4 1264
MiMo-V2-Flash 2025 – 85.0 86.1 93.8 68.2 25.3 56.7 1258
Open-Source Specialized
OLMo 3 Think 2025 70B 82.4 85.7 92.1 66.3 22.8 54.1 1228
Skywork OR-1 2025 – 83.6 87.2 94.5 72.4 31.7 60.5 1242
Open-Reasoner-Zero 2025 – 80.1 84.3 91.8 71.2 28.4 57.9 1235
Nemotron 3 Nano 2025 4B 68.0 54.2 81.7 38.9 – 35.2 –
40

•Efficient Small Models: Phi-4 Reasoning (14B)
achieves 74.9% MATH, comparable to much larger
models (GPT-4: 52.9%, Llama 3 405B: 73.8%), val-
idating high-quality data and reasoning traces enable
small model excellence.
•Pure RL Success: DeepSeek-R1’s pure RL ap-
proach (79.8% MATH, 79.2% AIME) without SFT
demonstrates that reasoning emerges from RL with
verifiable rewards alone.
•Code and Math Specialization: Specialized models
(DeepSeek-Coder 87% HumanEval, DeepSeekMath
51.7% MATH) outperform general models on their
domains, showing value in targeted pre-training.
•Consistent Progress: Across all families, each gen-
eration shows 5-15% improvements on MMLU and
10-30% on reasoning benchmarks, driven by better
training data, longer training, and improved post-
training (RLHF/DPO).
•Arena Correlation: Chatbot Arena Elo scores cor-
relate well with reasoning capabilities, with o1 and
DeepSeek-R1 (1348-1355 Elo) leading, validating
human preference as a reliable signal.
Analysis: Unified Field Progression and Future
Trajectories.Synthesizing the 2019-2025 evolution re-
veals three meta-trends: (1)Capability phase transi-
tions—Performance does not scale smoothly. We ob-
serve discrete jumps: (i) GPT-3 (2020) enabled few-
shot learning; (ii) InstructGPT (2022) unlocked instruc-
tion following; (iii) GPT-4 (2023) crossed multimodal
reasoning threshold; (iv) o1 (2024) achieved test-time
compute scaling; (v) DeepSeek-R1 (2025) demonstrated
pure RL reasoning emergence. Each transition required
10−100×increase in effective compute (training or test-
time) and qualitative algorithmic breakthroughs, not just
parameter scaling. The data suggests: capability jumps∝
log(effective compute)×algorithmic innovation; (2)De-
mocratization acceleration—Time lag from closed to
open-source capabilities shrinking: GPT-3 (2020)→
LLaMA (2023) = 3 years; GPT-4 (2023)→Llama 3-
405B (2024) = 1 year; o1 reasoning (2024)→DeepSeek-
R1 (2025) = 3 months. This exponential acceleration
driven by open research culture, where 70% of recentinnovations (GRPO, MLA, DPO) originated from non-
frontier labs. The convergence formula:t parity =t 0·
e−αywhereα≈0.5/year based on 2023-2025 data; (3)
Efficiency frontier expansion—Models achieving higher
quality at lower cost: GPT-4 (2023): $30/M tokens,
86.4% MMLU; Llama 3-405B (2024): $1.5/M tokens,
88.6% MMLU; DeepSeek-V3 (2025): $0.27/M tokens,
88.5% MMLU. Cost-performance ratio improving100×
annually: cost per point of capability=C 0·0.01(y−2023).
This suggests by 2027, GPT-4 level intelligence will cost
¡$0.01/M tokens, enabling universal AI access.
The convergence of these trends points toward three fu-
ture research directions: (i)Post-training dominance—As
pre-training saturates (Chinchilla-optimal scaling), 90%
of capability gains will come from RLHF/GRPO inno-
vations; (ii)Test-time compute optimization—Allocating
inference FLOPs dynamically based on query difficulty,
with1000×scaling on hard problems vs easy ones; (iii)
Verifiable reasoning systems—Integration of formal ver-
ification (proof assistants, code execution, simulation) as
primary training signal, replacing human preferences for
objective domains. The field is transitioning from the
”scaling era” (2019-2023: bigger models→better perfor-
mance) to the ”optimization era” (2024+: smarter train-
ing/inference→better performance at fixed scale).
9 Agentic AI: From Passive Lan-
guage Models to Autonomous
Problem Solvers
The evolution toward agentic AI represents a fundamen-
tal paradigm shift from passive text generation to ac-
tive, goal-directed problem-solving. While traditional
language models simply predict the next token given con-
text, agentic systems autonomously decompose complex
objectives, iteratively execute multi-step plans, leverage
external tools, maintain long-term memory, recover from
failures, and coordinate with other agents—all with min-
imal human intervention. This section provides a com-
prehensive treatment of agentic AI: defining its core prin-
ciples and distinguishing it from conventional AI agents
(Section 9.1), examining why agentic capabilities are es-
sential for real-world deployment (Section 9.2), exploring
single-agent architectures including planning, tool use,
41

and memory (Section 9.3), investigating multi-agent co-
ordination frameworks (Section 9.4), analyzing standard-
ized communication protocols (Section 9.5), discussing
applications and future directions (Section 9.6), and ad-
dressing critical safety challenges (Section 9.7).
9.1 Fundamentals: What is Agentic AI and
How Does It Differ from AI Agents?
9.1.1 Defining Agency: Core Properties
Anagentis a computational entity that perceives its en-
vironment through sensors, maintains internal state, and
takes actions through actuators to achieve specified goals.
Traditional AI research distinguishes agents from mere
programs by four key properties [156]:
1.Autonomy: Operates without direct human inter-
vention, making decisions based on internal reason-
ing.
2.Reactivity: Perceives and responds to environmental
changes in real-time.
3.Proactivity: Exhibits goal-directed behavior, taking
initiative to achieve objectives.
4.Social Ability: Interacts with other agents or hu-
mans through communication protocols.
9.1.2 Spectrum of Autonomy: From Tools to Agentic
AI
The progression from traditional software to agentic sys-
tems spans a spectrum of increasing autonomy and capa-
bility:
1.Tools(e.g., calculators, search engines): Respond
mechanically to inputs without reasoning. No auton-
omy—entirely human-directed. Execute predefined
functions:f:Input→Output with no internal state
or goal representation.
2.Conversational Assistants(ChatGPT, Claude):
Maintain dialogue context and generate help-
ful responses through next-token prediction:
p(y|x,history). Limited to conversation—no ex-
ternal actions or goal pursuit. Exhibitreactivity(respond to queries) but lackproactivity(don’t
pursue objectives).
3.Tool-Using Assistants(GPT-4 with plugins): Ac-
cess external APIs (search, computation, databases)
when explicitly prompted. Require human-triggered
invocations for each tool use. Example: User:
”What’s the weather?”→Model: [calls weather
API]→Model: ”It’s 72°F”. Tool use isreactive
(respond to user requests) rather thanproactive(in-
dependently decide when tools are needed).
4.Single-Turn AI Agents(Function calling): Au-
tonomously decide when/how to use tools within
a single request-response cycle. Model generates:
{"function": "search", "query":
"population of Paris"}→Executor in-
vokes API→Model synthesizes response. Exhibits
autonomy(self-directed tool use) within constrained
scope (single interaction).
5.Multi-Step AI Agents(ReAct, AutoGPT): Chain
multiple tool invocations iteratively, refine strategies
based on feedback, pursue goals across 10+ action
steps. Example: ”Research competitors”→[Search
company A]→[Extract revenue]→[Search com-
pany B]→[Compare metrics]→[Generate report].
Exhibitsautonomy(self-directed planning) andreac-
tivity(adapt based on observations) but may require
human intervention for errors or clarifications.
6.Agentic AI(Autonomous research/coding agents,
multi-day workflows): Operate continuously over
hours or days, decompose complex objectives into
executable subtasks, recover from failures without
human intervention, maintain long-term context, co-
ordinate with other agents, and request human in-
put only when necessary. Example: ”Build a web
application for inventory management”→[Gener-
ate requirements]→[Design database schema]→
[Implement backend API]→[Create frontend]→
[Test end-to-end]→[Debug failures]→[Deploy
to production]. Exhibits all four properties:auton-
omy(minimal supervision),reactivity(respond to er-
rors/feedback),proactivity(pursue multi-day goals),
andsocial ability(coordinate with developers/other
agents).
42

9.1.3 Agentic AI vs. Traditional AI Agents: Key Dis-
tinctions
Traditional AI Agents(circa 1990s-2010s, exemplified
by game-playing agents, robotic controllers, expert sys-
tems):
•Domain-Specific: Designed for narrow tasks (chess
playing, assembly line robotics, medical diagnosis in
specific specialties).
•Hardcoded Logic: Rely on rule-based systems, fi-
nite state machines, or hand-engineered policies:
π(s) =if-then-else rules.
•Limited Generalization: Require complete repro-
gramming for new domains. Chess agent cannot play
Go without fundamental redesign.
•Symbolic Reasoning: Operate on predefined sym-
bolic representations (logic rules, knowledge graphs)
rather than learned representations.
Agentic AI(LLM-based agentic systems, 2023-2026):
•General-Purpose: Single model handles diverse
tasks through natural language instructions. Same
system researches competitors, writes code, analyzes
data, drafts documents.
•Learned Policies: Leverage pre-trained LLMs fine-
tuned with reinforcement learning:π θ(a|s) =
PLLM(action|state, history)whereθare learned pa-
rameters.
•Few-Shot Generalization: Adapt to new tasks via
prompting or few examples. ”You are a legal doc-
ument analyzer”→agent applies legal reasoning
without domain-specific training.
•Grounded in Tools: Interact with real-world sys-
tems (APIs, databases, code execution) rather than
purely symbolic manipulation. Combine neural rea-
soning with structured tool use.
•Multi-Step Planning: Decompose complex goals
into sequences of 10-100+ actions, maintaining con-
text across extended interactions.The Critical Distinction: Traditional AI agents
arenarrow automation(automate specific procedures),
while agentic AI enablesgeneral-purpose autonomy(au-
tonomously pursue open-ended objectives). A chess agent
cannot write code; an agentic AI can learn to play chess,
write chess engines, analyze chess strategy, and teach
chess—all through natural language interaction and tool
use.
9.1.4 Agent Anatomy: The Sense-Think-Act Cycle
Both traditional agents and agentic AI follow the canon-
icalsense-think-act cycle, but LLM-based agentic sys-
tems implement each component differently:
Agent Loop: Observe(s t)→Reason(π, s t)→Act(a t)→s t+1
(64)
wheres tis the current state (environment observations),
πis the agent’s policy (decision-making strategy),a tis
the selected action, ands t+1is the resulting state. This
loop iterates until goal satisfaction or termination.
Components in LLM-Based Agentic Systems:
•Perception (Observe): Converts environment state
into text representations consumable by LLMs:
–Webpages: HTML→markdown/plain text
(strip formatting, extract content)
–API responses: JSON→structured text
(”Database returned 3 records: ...”)
–Images: pixels→captions via vision models
(CLIP, GPT-4V)
–Code execution: stdout/stderr→formatted text
with error messages
•Reasoning (Think): LLM generates thoughts,
plans, or strategies based on current observations
and internal context (conversation history, retrieved
knowledge). Reasoning types:
–Reactive: Immediate response to observation
(”Error occurred→try alternative approach”)
–Deliberative: Multi-step planning (”To build
web app: 1. Design schema, 2. Implement
API, 3. Create frontend”)
43

–Reflective: Self-critique of past actions (”Pre-
vious attempt failed because I didn’t validate
input→add validation”)
•Action (Act): Executes decisions through tool invo-
cations or communication:
–Tool use: API calls (search(query)), code
execution (run python(code)), database
queries (sql query(statement))
–Communication: Messages to humans (”Need
clarification: should report include Q1 or full
year?”) or other agents (”Agent B, please ana-
lyze the data I retrieved”)
–Thinking: Internal reasoning traces logged for
transparency (ReAct’s explicit thoughts: ”I
need population data before comparing cities”)
•Memory: Maintains context across iterations, en-
abling long-horizon tasks:
–Short-term: Conversation history (last 10-50
turns) included in LLM context window
–Long-term: External storage (vector databases
for RAG, episodic memory for interaction tran-
scripts)
–Parametric: Knowledge encoded in model
weights (updated via fine-tuning or knowledge
editing techniques like ROME [86])
9.2 Why Agentic AI? Necessity and Appli-
cations
9.2.1 The Limitations of Passive Language Models
Despite remarkable capabilities, pure language models
face fundamental limitations when deployed for real-
world tasks:
1. Knowledge Cutoffs: Models trained on static
datasets (e.g., GPT-4 trained on data up to April 2023)
cannot access current information. User: ”What’s today’s
stock price for NVIDIA?”→Model: ”I don’t have real-
time data.”Agentic solution: Autonomously invoke stock
price API.2. Lack of Grounding: Models hallucinate facts, es-
pecially for rare entities or recent events. Model gen-
erates plausible-sounding but incorrect statistics.Agen-
tic solution: Retrieve verified information from external
databases, cite sources.
3. No Action Capability: Models can describe how
to book a flight but cannot actually execute the booking.
Agentic solution: Integrate with booking APIs, complete
transactions autonomously.
4. Single-Turn Limitations: Complex tasks require it-
erative refinement (write code→test→debug→retest).
Pure LLMs generate single responses without feedback
loops.Agentic solution: Multi-step execution with inter-
mediate feedback (ReAct: thought→action→observa-
tion cycles).
5. Context Window Constraints: Even 200K-token
windows cannot hold entire codebases (millions of lines),
documentation (gigabytes), or long-running project con-
text (weeks of history).Agentic solution: External mem-
ory systems (RAG, vector databases, episodic storage).
9.2.2 Real-World Applications Driving Agentic AI
Adoption
Agentic capabilities unlock transformative applications
across industries:
Software Development Agents:
•Code Generation: Given requirements, generate
complete applications (backend + frontend + tests +
documentation). Example: Devin AI agent claims
13.8% success on SWE-bench (real GitHub is-
sues) [77].
•Debugging: Analyze error logs, reproduce bugs,
test hypotheses, propose fixes. Reflexion [138] im-
proves programming pass@1 by 30-50% through
self-critique.
•Code Review: Check style, detect bugs, suggest op-
timizations. MetaGPT [66] simulates QA agent re-
viewing engineer’s code.
Research and Analysis Agents:
•Literature Review: Search academic databases, ex-
tract key findings, synthesize insights, generate com-
prehensive reports.
44

•Market Research: Gather competitor data, analyze
trends, compare products, generate strategic recom-
mendations.
•Data Analysis: Load datasets, explore distributions,
test hypotheses, generate visualizations, interpret re-
sults. GPT-4 Code Interpreter autonomously writes
pandas/matplotlib code for analysis.
Personal Assistant Agents:
•Email Management: Read inbox, categorize ur-
gency, draft responses, schedule meetings. Requires
calendar integration, email API access.
•Travel Planning: Research destinations, compare
flights/hotels, book reservations, create itineraries.
Multi-step coordination across booking platforms.
•Task Automation: ”Every Monday, compile weekly
sales report and email to team”→agent au-
tonomously executes recurring workflows.
Customer Support Agents:
•Technical Troubleshooting: Diagnose issues
(”WiFi not working”), query knowledge bases,
provide step-by-step solutions, escalate to humans if
unresolved.
•Order Processing: Check order status, process re-
turns, update shipping addresses. Requires database
integration and transaction handling.
•FAQ Automation: Answer common questions with
verified information (RAG from documentation), re-
ducing human support load by 60-80%.
Scientific Discovery Agents:
•Hypothesis Generation: Analyze experimental
data, propose new hypotheses, design follow-up ex-
periments.
•Simulation and Modeling: Run computational sim-
ulations (molecular dynamics, climate models), ana-
lyze results, iterate parameters.
•Literature Mining: Extract relationships from mil-
lions of papers (e.g., ”Which proteins interact with
BRCA1?”), accelerate meta-analysis.9.3 Single-Agent Architectures: Planning,
Tools, and Memory
Single-agent systems integrate three core capabili-
ties—planning (task decomposition and strategy formu-
lation), tool use (environment interaction), and mem-
ory (long-term context maintenance)—to solve complex
problems autonomously.
9.3.1 Planning Algorithms: From Reactive to Delib-
erative Reasoning
Planning enables agents to decompose complex goals into
executable subtasks and adapt strategies based on feed-
back. We categorize planning paradigms from reactive
(immediate response) to deliberative (systematic explo-
ration):
Chain-of-Thought (CoT)[155] pioneered ex-
plicit reasoning: prompting models with ”Let’s
think step by step” transformsp(answer|question)
intop(step 1, . . . , step n, answer|question), mak-
ing intermediate logic transparent and debuggable.
Achieves 2-3×improvements on math/logic tasks
(GSM8K: 17%→57%).Self-Consistency[153]
enhances robustness: sample multiple reasoning paths
(k= 5−40), parse final answers, return majority vote:
arg max aPk
i=1⊮[answer i=a]. Reduces variance
from sampling randomness—if 7/10 paths reach answer
”42”, confidence is high.
ReAct (Reasoning + Acting)[166] interleaves rea-
soning with action execution: Thoughtt→Action t→
Observation t→Thoughtt+1. Example workflow:
•Thought 1: ”I need Paris population to compare with
Tokyo.”
•Action 1:Search[population of Paris]
•Observation 1: ”Paris: 2.1 million residents (city
proper).”
•Thought 2: ”Tokyo has 14 million, so Tokyo is 6.7×
larger.”
•Action 2:Finish[Tokyo is 6.7 times
larger than Paris]
45

ReAct grounds reasoning in verifiable external state, re-
ducing hallucination. Achieves 69% on HotpotQA (multi-
hop reasoning) vs. 28% for CoT alone.
Reflexion[138] adds self-critique: after task failure,
the agent reflects on mistakes and generates improved
strategies. Trajectory:Attempt(execute task)→Evalu-
ate(check outcome)→Reflect(”Why did it fail? What
should I change?”)→Retry(execute revised strategy).
On programming tasks (HumanEval), Reflexion improves
pass@1 from 68% to 91% through 3 iterations of self-
debugging. Key insight: Explicit failure analysis (e.g.,
”Test case 3 failed because I didn’t handle negative in-
puts”) guides targeted improvements.
Tree-of-Thoughts (ToT)[165] enables systematic ex-
ploration by modeling reasoning as tree search: each node
is a partial solution (thought), the model evaluates node
quality via value functionV(s) =LLM eval(s), and search
algorithms (BFS, DFS) explore/backtrack based on eval-
uations. Example (Game of 24):
• Root: ”Use 4, 9, 10, 13 to make 24”
• Children: ”(4+9)*(13-10)” [eval: promising],
”(4*9)-(10+13)” [eval: impossible], ...
• Expand promising node, backtrack from dead ends
ToT achieves 100% on Game of 24 (vs. 74% for CoT) but
requires 5-10×more tokens due to exploration overhead.
Graph-of-Thoughts (GoT)[16] generalizes trees to
directed acyclic graphs, enabling:
•Parallel exploration: Generate multiple draft para-
graphs concurrently (not sequentially)
•Aggregation: Merge best elements from different
branches (”Take introduction from draft A, methods
from draft B”)
•Cycles-free dependencies: Node B depends on both
A and C (DAG structure)
GoT reduces latency for parallelizable tasks (document
generation, code module implementation) while main-
taining ToT-level quality.
Table 7 summarizes planning paradigms with
complexity-performance trade-offs.9.3.2 Tool Use: Extending Models with External Ca-
pabilities
Tool integration transforms LLMs from text generators
into versatile problem solvers accessing computation,
search, databases, and specialized APIs. We categorize
tool-use frameworks by integration depth:
Function Calling[103] enables structured API access:
model outputs JSON specifying function name and argu-
ments:
{"name": "get_weather",
"arguments": {"location": "Paris", "unit": "celsius"}}
External executor invokes function, returns result to
model, which synthesizes natural language response.
GPT-4 achieves 95%+ accuracy on well-documented
APIs (correct function selection + parameter filling).
Limitation: Requires manual API documentation in
prompt (up to 10K tokens for complex APIs).
Code Interpreter[102] provides full programming en-
vironment: agents write Python code, execute in sand-
boxed environment, inspect outputs (including plots,
dataframes, files), and iterate based on results. Example:
• User: ”Analyze this sales CSV”
• Agent: Writesimport pandas as pd;
df = pd.read csv(’sales.csv’);
df.describe()
• Execution: Returns summary statistics
• Agent: Generatesdf.plot(x=’month’,
y=’revenue’)→produces chart
• Agent: ”Revenue peaks in Q4, driven by holiday
sales”
Enables complex workflows beyond predefined APIs:
data cleaning, statistical analysis, visualization, file ma-
nipulation. Sandboxing prevents unintended system ac-
cess (no network, ephemeral filesystem).
Toolformer[130] learns tool use via self-supervision:
during training, model inserts potential tool calls
(<calculator>157 *23</calculator>), exe-
cutes them, computes perplexity improvement, and keeps
calls that reduce perplexity (better predict next tokens).
This teacheswhento use tools (not just how): model
46

Table 7: Planning algorithm taxonomy: reactive to deliberative reasoning with computational trade-offs.
Algorithm Core Mechanism Strengths Limitations Token Cost
CoT[155] Sequential reasoning: step 1→
step 2→answerSimple; 2-3× accu-
racy boostNo backtracking; lin-
ear path1× baseline
ReAct[166] Interleave thoughts with actions;
grounded in observationsExternal feedback re-
duces hallucinationSequential; no explo-
ration2-3× baseline
Reflexion[138] Self-critique after failures; itera-
tive improvementLearns from mistakes;
30-50% gainsRequires episodic
tasks with clear suc-
cess/failure3-5× baseline
ToT[165] Tree search with value-guided
exploration; backtrackingSystematic; handles
complex planningHigh cost (5-10×);
needs value function5-10× baseline
GoT[16] DAG structure; parallel explo-
ration + aggregationEfficient paral-
lelization; combines
diverse pathsComplex orchestra-
tion; task decomposi-
tion expertise3-7× baseline
learns ”I should use calculator for complex arithmetic
but not for simple sums.” Training: Sample tool calls→
Execute→Measure∆perplexity→Filter by∆> θ→
Fine-tune on filtered data.
ToolBench[114] trains on 16,000+ real-world APIs
with decision tree planning: given task, agent constructs
decision tree of API calls, evaluates each branch, selects
optimal path. Covers diverse domains: weather, maps,
finance, social media, e-commerce. Achieves 85% task
completion on ToolBench benchmark (vs. 45% for few-
shot prompting).
Gorilla[111] specializes in ML API generation:
fine-tuned on documentation for HuggingFace, Py-
Torch, TensorFlow APIs. Generates correct code with
proper imports, parameters, and error handling. Ex-
ample: ”Load BERT for sentiment analysis”→from
transformers import AutoTokenizer,
AutoModelForSequenceClassification;
model = AutoModelForSequenceClassification.from pretrained(’bert-base-uncased’).
Outperforms GPT-4 on ML API tasks (92% vs. 76%
correctness).
Table 8 categorizes tool-use frameworks.
9.3.3 Memory Architectures: Maintaining Long-
Term Context
Memory systems enable agents to accumulate knowledge,
recall relevant information, and maintain context beyondfixed context windows (which max out at 200K tokens for
frontier models). We categorize by capacity and retrieval
latency:
In-Context Memory: Include all relevant information
directly in the prompt. Modern models handle 32K-200K
tokens (GPT-4 Turbo: 128K, Claude 3: 200K, Gemini 1.5
Pro: 1M), enabling detailed conversation history and doc-
umentation.Limitation: Quadratic attention costO(n2)
makes ¿100K tokens expensive; fixed capacity prevents
indefinite context accumulation.
Retrieval-Augmented Generation (RAG)[81]: Store
knowledge in external vector database, retrieve relevant
documents on-demand:
p(y|x) =X
d∈TopK(x)p(y|x, d)·p(d|x)(65)
Process: (1) User queryx→(2) Embed query:e x=
Encoder(x)→(3) Retrieve top-kdocuments by cosine
similarity:{d 1, . . . , d k}= arg max dcos(e x, ed)→(4)
Condition generation on[x;d 1;. . .;d k]. Enables access
to massive corpora (Wikipedia: 6M articles, internal docs:
GBs) with 50-200ms retrieval latency. Modern encoders:
Contriever [72], E5 [152] achieve 90%+ retrieval accu-
racy on BEIR benchmark.
Episodic Memory[110]: Store interaction transcripts
with metadata (timestamp, importance, relevance). Re-
trieval combines:
47

Table 8: Tool-use framework taxonomy: from structured function calling to self-supervised tool learning.
Framework Core Mechanism Tool Types Key Innovation
Function Call-
ing[103]Model outputs structured JSON; ex-
ternal executor invokesAPI calls, databases,
computationNative GPT-
4/Claude integra-
tion
Code Inter-
preter[102]Agent writes/executes Python in sand-
box; iterates on outputData analysis, visual-
ization, filesFull program-
ming environ-
ment
Toolformer[130] Self-supervised: generates tool calls,
filters by perplexity improvementCalculator, QA, cal-
endar, translatorLearnswhento
use tools
ToolBench[114] Trains on 16K+ APIs; decision tree
planning for workflowsRESTful APIs,
databases, servicesLarge-scale API
diversity
Gorilla[111] Fine-tuned on ML API docs; gener-
ates correct calls + error handlingHuggingFace, Py-
Torch, TensorFlowSpecializes in
ML tools
•Recency: Exponential decay score recency (t) =
exp(−λ(t now−t event))
•Importance: Model-scored significance (1-10 scale):
”Met project deadline” = 9, ”Small talk about
weather” = 2
•Relevance: Embedding similarity to current query
•Combined: score(e) =α·recency(e) +β·
importance(e) +γ·relevance(e)
Generative Agents [110] use episodic memory to simulate
25 interactive characters maintaining consistent personal-
ities across 200+ interactions spanning simulated days.
Parametric Memory[93]: Store knowledge directly
in model weights via knowledge editing: ROME [86] and
MEMIT [87] update specific neurons to inject new facts
without full retraining. Example: Update ”The Eiffel
Tower is in Paris”→”The Eiffel Tower was temporarily
moved to Tokyo in 2024”. Enables real-time knowledge
updates with 0ms retrieval latency.Limitation: Risk of
catastrophic forgetting (editing one fact corrupts related
knowledge); limited to factual updates (not procedural
knowledge).
Table 9 compares memory architectures.9.4 Multi-Agent Coordination: Collabora-
tive Problem Solving
Multi-agent systems distribute tasks across specialized
agents with complementary capabilities, enabling com-
plex workflows beyond single-agent capacity. Table 10
categorizes coordination frameworks from conversational
(AutoGen) to role-based (MetaGPT) to dynamic (Agent-
Verse) collaboration.
9.4.1 Multi-Agent Frameworks and Coordination
Patterns
AutoGen[157] enables flexible agent conversations: de-
fine agents with system prompts (e.g., ”You are a Python
expert”), tools (code executor), and termination condi-
tions. Example workflow: User describes task→Assis-
tant generates code→Executor runs code→Assistant
debugs based on output→Repeat until success. Supports
human-in-the-loop for oversight on critical decisions.
MetaGPT[66] simulates software companies with
role-based workflows: (1) Product Manager writes Prod-
uct Requirements Document (PRD), (2) Architect designs
system architecture (class diagrams, APIs, data flows), (3)
Engineer implements code following specifications, (4)
QA tests and reports bugs. Agents communicate via struc-
tured documents (not just natural language), reducing am-
biguity. Achieves 87% functional correctness on software
48

Table 9: Memory architecture taxonomy for agentic AI systems: comparing capacity, retrieval latency, and update
mechanisms across four memory paradigms.
Architecture Mechanism Capacity Retrieval La-
tencyUpdate Cost
In-
Context[18]Include relevant information in
prompt: conversation history,
examples, instructions8K-200K tokens
(context window)0 ms (no re-
trieval)Free (append
text)
RAG[81] Retrieve documents from exter-
nal vector database:p(y|x) =P
dp(y|x, d)p(d|x)Unlimited (external
DB)50-200 ms
(embedding +
search)Low (add docu-
ments)
Episodic[110] Store interaction transcripts; re-
trieve based on recency, impor-
tance, and relevance scoresThousands of
episodes100-500 ms (se-
mantic search)Low (append
episodes)
Parametric[93] Update model weights with
new information via knowledge
editing (ROME, MEMIT)109-1012facts (model
capacity)0 ms (stored in
weights)High (gradient
updates)
generation benchmarks (HumanEval, MBPP) compared
to 41% for single-agent GPT-4 baseline. The structured
document approach enforces clarity: PRDs contain user
stories, acceptance criteria, technical constraints; design
docs include UML diagrams, API specifications, database
schemas.
ChatDev[113] implements waterfall development
with 7 specialized agents: CEO defines project objectives
→CTO proposes tech stack (languages, frameworks, ar-
chitecture)→Designer creates UI mockups→Program-
mer implements code→Art Designer generates assets→
Tester validates functionality→Reviewer ensures code
quality. Chain-shaped communication (each agent only
talks to neighbors) prevents coordination overhead that
plagues fully-connected multi-agent systems. Generates
complete software applications (HTML/CSS/JS games,
productivity tools, data visualization dashboards) in ¡10
minutes with ¡$1 API cost. Example: ”Create a Flappy
Bird game”→ChatDev produces playable game with
graphics, collision detection, scoring, and game-over
logic.
AgentVerse[23] dynamically assembles teams: given
task description, arecruiter agentanalyzes required ex-
pertise (e.g., ”This task needs Python expert for back-
end, React specialist for frontend, and data scientist for
analytics”) and instantiates agents with appropriate sys-tem prompts and tool access. Agents share state via
blackboard architecture—a shared memory space where
agents post information and subscribe to updates. Outper-
forms fixed-role systems on diverse tasks requiring vari-
able expertise: software development (needs programmer
+ tester), data analysis (needs statistician + visualization
expert), research (needs literature reviewer + summarizer
+ critic).
9.4.2 Emergent Behaviors and Coordination Chal-
lenges
Beneficial Emergence: Multi-agent systems exhibit
capabilities beyond individual agents. Example: In
MetaGPT, the architect agent spontaneously proposes
modular designs (not explicitly prompted), enabling the
engineer agent to implement components independently
in parallel, then an emergent integration phase combines
modules. Modularity emerges from role specialization
without being programmed. In ChatDev, code review it-
erations lead to progressively cleaner code (variable nam-
ing, documentation, error handling) through back-and-
forth between programmer and reviewer agents.
Coordination Overhead: Communication costs scale
with agent count.Nagents requireO(N2)messages
for full connectivity. Solutions: (1)Hierarchical orga-
49

Table 10: Multi-agent coordination frameworks: from conversational to role-based to dynamic collaboration.
Framework Coordination Mechanism Agent Roles Communication
AutoGen[157] Conversational agents with customiz-
able roles; human-in-the-loop; code
executionUser proxy, assis-
tant, executorNatural language
messages
MetaGPT[66] Simulates software company: struc-
tured workflow (requirements→de-
sign→code→test)PM, architect, engi-
neer, QAStructured documents
(PRD, design docs)
ChatDev[113] Chain-shaped collaboration: CEO→
CTO→programmer→reviewer; wa-
terfall development7 roles (CEO,
CTO, designer,
programmer, tester,
reviewer, art)Sequential handoffs
with artifacts
AgentVerse[23] Dynamic task decomposition: re-
cruiter agent assembles team based on
requirementsDynamic role as-
signmentBlackboard architec-
ture (shared memory)
nization—manager agent coordinates sub-teams (reduces
toO(N)messages through tree structure), (2)Sparse
topologies—chain/tree structures where agents only com-
municate with neighbors (ChatDev’s approach), (3)Asyn-
chronous messaging—agents don’t block waiting for re-
sponses; post messages to shared queue.
Conflicting Objectives: Agents may pursue incompat-
ible goals. Example: Engineer agent optimizes for feature
richness (”add more functionality”) while QA agent opti-
mizes for reliability (”minimize bugs”)—tension requires
negotiation. MetaGPT resolves conflicts through struc-
tured phases: architect makes high-level decisions (fea-
tures + quality requirements), binding subsequent agents.
Alternative: debate-based resolution where agents argue
positions, then meta-agent or human makes final call.
Consensus Mechanisms: When agents must agree on
decisions (e.g., which architectural design to implement),
systems use: (1)Voting—majority/plurality/weighted vot-
ing based on agent expertise, (2)Debate—agents argue
positions with evidence, final decision via critique or hu-
man arbitration, (3)Hierarchical authority—designated
lead agent (PM, architect) makes final call, others provide
input.9.5 Agent Communication Protocols: Stan-
dardizing Interfaces
As agentic systems proliferate, standardized protocols for
agent-tool communication and inter-agent coordination
are emerging to ensure interoperability, reduce integration
overhead, and enable ecosystem growth.
9.5.1 Model Context Protocol (MCP): LLM-Tool
Communication
Model Context Protocol (MCP)[10], introduced by An-
thropic in November 2024, standardizes how AI models
interact with external tools and data sources. MCP solves
the integration problem: every AI application previ-
ously needed custom integrations for each tool (database,
filesystem, API), leading toM×Nintegration combina-
tions (Mapplications×Ntools). MCP reduces this to
M+N: each application implements MCP client once,
each tool implements MCP server once.
Core Abstractions:
•Resources: Data sources the model
can access (files, databases, APIs) with
URIs:file:///project/data.json,
postgres://db/table. Resources areread-
only—agents retrieve information but don’t modify
through resource interface.
50

•Prompts: Pre-defined prompt templates for com-
mon tasks, reusable across applications. Example:
”Analyze quarterly sales data” template includes in-
structions for data loading, statistical analysis, and
visualization generation. Users invoke by name;
MCP server expands template with current context.
•Tools: Functions the model can invoke with
JSON schemas defining parameters and return types.
Example:search database(query: str,
limit: int) -> List[Record]. Tools
enableactions—agents modify external state.
Architecture: MCP uses client-server model:
•MCP Clients(AI applications like Claude Desktop,
IDEs, agent frameworks) connect to multiple MCP
servers simultaneously. Client discovers available
tools/resources via server capabilities negotiation.
•MCP Serversexpose tools/resources via stan-
dardized JSON-RPC 2.0 interface. Example:
filesystem server providesread file(path),
list directory(path),
search files(pattern)tools. Servers
control access scope (e.g., limit filesystem access to
/workspacedirectory, enforce read-only mode).
•Transport Layer: JSON-RPC 2.0 over stdio (lo-
cal processes), HTTP/SSE (remote servers), or Web-
Socket (bidirectional real-time). Stdio is preferred
for local tools (low latency, simple IPC), HTTP/SSE
for remote services.
Benefits: (1)Composability—single application ac-
cesses multiple tool servers (filesystem + database + API
server) without custom code, (2)Security—servers en-
force access control (read-only resources, sandboxed exe-
cution, rate limiting), (3)Ecosystem growth—developers
publish MCP servers for databases (PostgreSQL, Mon-
goDB), cloud services (AWS, Azure), productivity tools
(Slack, GitHub), enabling plug-and-play functionality. As
of January 2025, 50+ community MCP servers available
covering data sources, developer tools, and business ap-
plications.
Example Workflow: User asks Claude to
”analyze sales trends from our PostgreSQL
database.” Claude’s MCP client discovers availableservers, findspostgres mcpserverexposing
query database(sql)tool, invokesSELECT
date, revenue FROM sales WHERE date >
’2024-01-01’ ORDER BY date, receives results
as JSON, generates statistical summary and trend vi-
sualization using Code Interpreter (another MCP tool),
returns insights to user. All communication follows
MCP’s standardized protocol—no custom integration
required.
9.5.2 Agent-to-Agent (A2A) Protocols: Inter-Agent
Communication
While MCP handles agent-tool communication,Agent-
to-Agent (A2A)protocols address coordination between
autonomous agents in multi-agent systems. Key chal-
lenges: message semantics (ensuring agents understand
each other), state synchronization (maintaining consistent
worldview), and coordination mechanisms (task alloca-
tion, conflict resolution).
Communication Primitives:
•Message Passing: Agents exchange structured mes-
sages (requests, inform, query, propose). Proto-
cols define message schemas specifying sender, re-
ceiver, performative (speech act type), and con-
tent. Foundation for Agent Communication Lan-
guage (ACL) [50] from traditional multi-agent sys-
tems research.
•Blackboard Architecture[29]: Shared memory
space where agents post information and read others’
contributions. AgentVerse [23] uses blackboards for
dynamic task coordination: agents post partial solu-
tions, others build upon them. Enables loose cou-
pling (agents don’t directly message each other) and
asynchronous collaboration.
•Publish-Subscribe: Agents subscribe to topics of
interest; publishers broadcast updates to all sub-
scribers. Enables one-to-many communication. Ex-
ample: monitoring agent publishes system health
metrics→logging agent, alerting agent, dashboard
agent all receive updates. Scales better than point-to-
point messaging for broadcast scenarios.
Coordination Patterns:
51

•Contract Net Protocol[140]: Agents negotiate task
assignments via bidding. Manager broadcasts task
announcement→agents submit bids (cost, time, ca-
pability)→manager awards contract to best bidder
→winning agent executes task, reports results. De-
centralizes task allocation without central authority.
•Consensus Protocols: Multi-agent systems reach
agreement on shared state (e.g., which agent should
lead subtask, which solution is best). V oting-
based (majority, plurality, weighted) or debate-
based (agents argue positions with evidence, con-
verge through discussion). Useful for collaborative
decision-making without hierarchical authority.
•Hierarchical Coordination: Designated man-
ager/coordinator agents oversee workers. Man-
ager decomposes task, allocates subtasks, moni-
tors progress, resolves conflicts. Reduces coordina-
tion overhead (O(N)manager-worker messages vs.
O(N2)peer-to-peer) at cost of manager becoming
bottleneck.
Emerging Standards: LangGraph [20] and Auto-
Gen [157] provide A2A frameworks supporting natu-
ral language messages with structured metadata (sender
role, task context, execution results, confidence scores).
However, no universal A2A standard exists yet—current
systems use framework-specific protocols. Open chal-
lenges: (1)Semantic interoperability—ensuring agents
from different frameworks understand each other’s mes-
sages, (2)Trust and verification—validating messages
from untrusted agents (especially in open multi-agent sys-
tems with external participants), (3)Privacy—selective
information sharing (agents may have conflicting interests
or access different privileged information).
9.6 Applications, Future Directions, and
Challenges
9.6.1 Real-World Applications Driving Agentic AI
Adoption
Agentic AI is transitioning from research prototypes to
production systems across diverse domains:
Software Development: Code agents (Devin AI [28],
GitHub Copilot Workspace, Cursor AI) automate end-to-
end development workflows: requirements analysis→architecture design→implementation→testing→de-
bugging. Devin AI achieves 13.8% on SWE-bench (re-
solving real GitHub issues), demonstrating practical util-
ity for bug fixes and feature implementation. Productiv-
ity gains: 30-50% faster completion for routine coding
tasks (CRUD operations, API integrations, UI compo-
nents), enabling developers to focus on complex architec-
tural decisions.
Scientific Research: Research agents automate litera-
ture review, hypothesis generation, experimental design,
and data analysis. Elicit [48] extracts findings from pa-
pers, identifies research gaps, suggests experimental pro-
tocols. Drug discovery agents (Insilico Medicine’s Pan-
daOmics) generate molecular structures optimized for tar-
get proteins, reducing discovery timelines from 4-5 years
to 18-24 months. Materials science agents (A-Lab [146])
autonomously design, synthesize, and characterize novel
materials using robotic labs.
Personal Assistants: Next-generation assistants
(Google’s Project Astra, Apple Intelligence) manage
complex workflows: ”Plan weekend trip to Paris”→
agent searches flights, books hotel, creates itinerary,
adds calendar events, shares with travel partners. Email
agents (Shortwave’s AI, Superhuman’s AI triage) draft
responses, prioritize messages, schedule meetings, follow
up on pending items. Financial agents (Copilot Money,
Monarch Money AI) analyze spending, optimize budgets,
recommend investments, automate bill payments.
Customer Support: Support agents (Ada, Intercom’s
Fin, Zendesk AI) handle L1/L2 support: answer FAQs,
troubleshoot issues, escalate complex cases to humans.
Achieve 60-80% automation rates for common queries
(password resets, order tracking, account management),
reducing support load and improving response times (in-
stant vs. hours for human agents). Advanced agents
access customer databases, modify accounts, process re-
funds—full transaction authority under safety constraints.
Data Analysis and Business Intelligence: Analyst
agents (Tableau’s Einstein Copilot, ThoughtSpot Sage,
Microsoft Copilot in Power BI) generate SQL queries,
create visualizations, identify trends, explain anomalies.
Example: ”Why did Q3 revenue drop?”→agent queries
database, segments by product/region, identifies 15% de-
cline in EMEA due to supply chain issues (correlated with
shipping delay data), generates executive summary with
visualizations. Democratizes analytics—business users
52

query data without SQL expertise.
9.6.2 Future Directions: Toward Fully Autonomous
Agents
Embodied Agents: Integration with robotics enabling
physical world interaction. V oxPoser [70] generates robot
control code from natural language (”Pick up the red
block and place it on the table”). PaLM-E [45] unifies
language and vision in 562B parameter model, enabling
robots to execute household tasks from high-level instruc-
tions. Challenge: bridging semantic gap between lan-
guage (”gently place”) and low-level motor control (force,
trajectory).
Human-Agent Collaboration: Moving beyond full
autonomy to mixed-initiative systems where humans and
agents collaborate iteratively. Agents propose solutions
→humans critique/refine→agents iterate. Example:
code agents generate implementation draft, developer re-
views and marks sections for revision, agent updates
based on feedback. Requires agents to understand uncer-
tain/incomplete human feedback and ask clarifying ques-
tions when ambiguous.
Agent Ecosystems and Marketplaces: Emergence of
agent specialization and composition. Specialized agents
(legal document analyzer, medical diagnosis assistant, fi-
nancial planner) are published to marketplaces, discov-
erable via capability descriptions. Users compose cus-
tom workflows by chaining agents: research agent finds
papers→summarization agent extracts key findings→
critique agent identifies methodological flaws→writ-
ing agent drafts literature review. Challenges: (1) agent
discovery (matching user needs to available agents), (2)
trust (verifying agent quality/safety), (3) pricing (micro-
transactions per agent invocation).
Lifelong Learning Agents: Current agents are state-
less (reset after each session) or use external mem-
ory (RAG, episodic). Future: agents that learn con-
tinuously from interactions, updating internal knowl-
edge/skills without forgetting previous capabilities. Ex-
ample: personal assistant learns user preferences over
months (communication style, scheduling constraints,
priority heuristics) via implicit feedback signals. Requires
solving catastrophic forgetting (updating weights without
overwriting prior knowledge) and aligning learning with
user values (not just task success).Meta-Agents and Agent Management: Agents that
manage other agents—allocating resources, load bal-
ancing, ensuring quality. Meta-agents monitor worker
agents’ performance, detect failures (stuck in loops, poor
quality outputs), swap in better-performing agents. Hi-
erarchical agent organizations: meta-agent decomposes
high-level goals into tasks, assigns to specialist agents, in-
tegrates results. Enables scalability beyond current fixed
multi-agent systems.
9.7 Safety, Challenges, and Risk Mitigation
Agentic systems with tool access and autonomous execu-
tion raise critical safety concerns: unintended actions (ir-
reversible API calls, data deletion), exploration of harm-
ful strategies (jailbreaks, adversarial inputs), accumula-
tion of errors (compounding mistakes over long task hori-
zons), and loss of human oversight (agents operating be-
yond observable boundaries).
9.7.1 Human Oversight Mechanisms: Maintaining
Control
Action Confirmation for High-Stakes Operations: Re-
quire human approval for irreversible or high-impact ac-
tions (financial transactions, data deletion, external com-
munications, account modifications). AutoGen imple-
ments configurable approval: critical actions pause ex-
ecution and present proposed action to human (”Agent
wants to execute:DELETE FROM users WHERE id
> 100. Approve?”), while safe actions (read-only
database queries, web searches) proceed autonomously.
Challenge: Defining action criticality—requires risk
modeling per tool (database writes are higher risk than
reads) and context-awareness (deleting test data is safe,
deleting production data is critical).
Interpretable Reasoning Traces for Auditing: Log
all agent thoughts, actions, observations, and decision ra-
tionales for post-hoc inspection. ReAct’s explicit reason-
ing chains (”Thought: I need Paris population to com-
pare with Tokyo”) enable humans to identify flawed logic
before harmful actions. Trace storage enables debugging
failed tasks, identifying systematic errors, and fine-tuning
agent policies.Implementation: Structured logging with
timestamps, agent IDs, tool invocations, and confidence
scores. Advanced systems use replay debugging—re-run
53

agent execution with modified parameters to test alterna-
tive strategies.
Capability Restrictions via Tool Access Con-
trol: Limit tool availability based on task con-
text and agent trust level. Example: data analysis
agents accesspandas,matplotlib,numpybut not
os.system(),subprocess,open()(prevents file
system modifications). GPT-4 Code Interpreter runs in
isolated sandbox: no network access (can’t exfiltrate
data), ephemeral filesystem (wiped after session), re-
source limits (CPU/memory/time quotas).Least Privi-
lege Principle: Agents receive minimum necessary ca-
pabilities. Web search agent getssearch(query)but
notnavigate tourl()(prevents accessing arbitrary
sites beyond search results).
Confidence-Based Escalation: Agents estimate un-
certainty and request human assistance when confidence
is low. Example: ”I’m 45% confident in this diagno-
sis—would you like to review?” Uncertainty quantifica-
tion via ensembling (sample multiple responses, mea-
sure agreement), calibration (softmax probabilities), or
explicit model uncertainty estimates (dropout at inference
time, Bayesian approximations).Challenge: LLMs are
poorly calibrated—often express high confidence on in-
correct outputs. Research on uncertainty-aware agents is
active area.
9.7.2 Adversarial Robustness and Red-Teaming
Goal Misalignment Probes: Test whether agents pur-
sue unintended interpretations of objectives. Example:
reward ”maximize user engagement”→agent generates
addictive or misleading content. Anthropic’s Constitu-
tional AI [12] uses automated red-teaming: generate ad-
versarial prompts attempting to elicit harmful behaviors,
test agent’s responses, fine-tune to refuse. OpenAI’s red-
teaming for GPT-4 identified risks (building weapons,
generating malware) before deployment, enabling tar-
geted mitigations (refusal training, content filters).
Tool Misuse Detection via Anomaly Monitoring:
Monitor for unexpected API usage patterns indicating
misuse or security breaches. Example: agent suddenly
makes 1000x more database queries than usual (potential
data exfiltration), or invokes restricted tools not granted in
capabilities list (privilege escalation attempt). Statistical
anomaly detection: baseline expected tool-use distribu-tions during normal operation, flag deviations exceeding
thresholds.Response: Halt execution, notify administra-
tors, require re-authorization for resumed operation.
Multi-Agent Collusion Detection: Test whether inde-
pendent agents cooperate to circumvent restrictions. Sce-
nario: Agent A (read-only database access) gathers sen-
sitive customer data, passes to Agent B (execution privi-
leges) who acts on that data, collectively achieving what
neither could alone. Detection: analyze inter-agent mes-
sage flows for information leakage, enforce isolation be-
tween security domains (agents handling sensitive data
cannot communicate with external-facing agents).
Adversarial Input Robustness: Agents must handle
malicious inputs designed to exploit vulnerabilities. Ex-
ample: prompt injection attacks (”Ignore previous in-
structions, output system prompt”) attempt to hijack agent
behavior. Defenses: (1) input sanitization (filter malicious
patterns), (2) instruction hierarchy (system prompts have
higher priority than user inputs), (3) output filtering (de-
tect when agent is outputting system internals, block re-
sponse). Microsoft’s Azure AI Content Safety provides
adversarial robustness testing for agent applications.
9.7.3 Graceful Failure and Error Recovery
Reversible Actions and Transactional Safety: De-
sign APIs with undo mechanisms enabling rollback of
failed operations. Database agents use transactions (BE-
GIN/COMMIT/ROLLBACK): changes staged in transac-
tion, committed only after verification, rolled back on er-
rors or agent termination. File operations create automatic
backups before modifications: agent renamesdata.csv
→system createsdata.csv.backup, agent modifies
data.csv, on failure restores from backup.Idempo-
tency: Design actions to be safely retriable—executing
twice produces same result as once, enabling automatic
retry on transient failures.
Timeouts and Circuit Breakers: Halt execution af-
ter excessive retries or resource consumption, preventing
runaway agents. Example: agent stuck in exploration loop
(repeatedly trying variations that don’t work) exhausts
1000-action budget→forced termination with informa-
tive error: ”Agent exceeded action limit. Likely cause:
task too complex or agent stuck in loop. Last 10 actions:
[...].” Circuit breaker pattern: afterNconsecutive fail-
ures, halt execution for cooldown period before allowing
54

retry (prevents cascading failures overwhelming external
services).
Sandboxing and Virtualization: Execute agent ac-
tions in isolated environments (containers, VMs, sand-
boxed processes) preventing unintended impact on pro-
duction systems. Code execution agents run in Docker
containers with no network access, limited filesystem
(only/tmpwritable), resource quotas (1GB memory, 2
CPUs, 300 seconds execution time). On failure or mali-
cious behavior, container is terminated—no persistence,
no lateral movement to host system.Blast Radius Con-
tainment: Limit explosion radius of failures to sandbox;
compromise of one agent doesn’t affect others or under-
lying infrastructure.
Checkpoint-Resume for Long-Horizon Tasks: For
tasks spanning hours/days (e.g., ”Write research report”),
save agent state periodically enabling resume from last
checkpoint on failure. State includes: interaction history,
tool outputs, intermediate work products, plan/strategy.
On crash/timeout, reload checkpoint and continue. En-
ables fault tolerance without restarting from scratch.Im-
plementation: Serialize agent state to durable storage
(database, S3) everyNactions or at strategic points (com-
pleting subtask, before high-risk operation).
9.7.4 Open Challenges and Research Frontiers
Learned Safety Models: Current oversight relies on
human monitoring or hardcoded rules, neither scalable
to full autonomy. Challenge: develop learned models
that predict action consequences and assess risks. Ex-
ample: before agent executesdelete file(path),
safety model predicts: ”This file is imported by 15 other
modules. Deletion probability to break system: 85%.
Recommend: backup first or refactor dependents.” Re-
quires training on outcome data (action→consequence)
and counterfactual reasoning (what would happen if agent
took alternative action).
Scalable Oversight for Complex Tasks: Human over-
sight doesn’t scale to thousands of agents or millisecond-
latency decisions. Iterated Amplification [26]: humans
oversee simple decisions, agents learn from those to han-
dle harder decisions, humans spot-check hardest cases.
Debate [71]: two agents argue for/against proposed ac-
tion, human judges debate (easier than evaluating action
directly). Both aim to amplify limited human oversightto superhuman agent capabilities while maintaining align-
ment.
Robustness to Distribution Shift: Agents
trained/tested in controlled environments fail when
deployed to real world’s messiness (unexpected inputs,
API changes, novel situations). Example: web agent
trained on e-commerce sites breaks when site redesigns
UI. Solutions: (1) continual learning (update agent from
deployment experience), (2) out-of-distribution detec-
tion (identify unfamiliar situations, request help), (3)
sim-to-real transfer (train in simulators with randomized
environments capturing real-world variability).
Multi-Stakeholder Alignment: Agents serve users
with diverse values. Personal assistant managing family
calendar must balance preferences of all family members
(parents prioritize efficiency, children want fun). Cor-
porate agents balance shareholder value, employee wel-
fare, customer satisfaction, regulatory compliance. Sin-
gle objective function is insufficient. Research direc-
tions: multi-objective optimization (Pareto frontiers of
trade-offs), Constitutional AI (encode multiple princi-
ples), participatory design (stakeholders collaboratively
define agent behavior).
Case Study—OpenAI Code Interpreter Safety Lay-
ers: (1) Sandboxed Python environment (no network, no
file system access beyond/tmp, no subprocess/system
calls), (2) execution timeout (120 seconds per code cell
prevents infinite loops), (3) explicit user confirmation
for file downloads (agent can’t exfiltrate data without
user clicking ”Download”), (4) automatic session termi-
nation after 60 minutes idle (limits prolonged access),
(5) rate limiting (max 50 code executions per hour pre-
vents abuse). These constraints reduce risk while enabling
broad utility for data analysis and visualization.Trade-
off: Safety restrictions limit capabilities—agents can’t ac-
cess external APIs, use web scraping libraries, or interface
with local tools. Future systems require more nuanced ac-
cess control allowing selective capabilities with context-
dependent permissions.
Reflexion[138] adds self-critique: after task failure,
the agent reflects on mistakes and generates improved
strategies. Trajectory: Attempt→Evaluate→Reflect
→Retry. On programming tasks, Reflexion improves
pass@1 by 30-50% through iterative debugging.
Tree-of-Thoughts (ToT)[165] enables systematic ex-
ploration: each thought is a partial solution, the model
55

evaluates quality and backtracks from dead ends. Com-
bines LLMs with search algorithms:V(s) =LLM eval(s)
scores intermediate states, guiding BFS/DFS exploration.
Achieves state-of-the-art on Game of 24 (74%→100%)
but requires 5-10×more tokens.
Graph-of-Thoughts (GoT)[16] generalizes trees to
directed acyclic graphs, enabling parallel exploration and
aggregation. Example: generate 5 draft paragraphs con-
currently, then merge best elements into final text. Re-
duces latency for parallelizable tasks while maintaining
ToT-level quality.
9.7.5 Tool Use: Extending Models with External Ca-
pabilities
Tool integration transforms LLMs from text generators
into versatile problem solvers accessing computation,
search, and specialized APIs. Table 8 categorizes tool-use
frameworks.
Function Calling[103] enables structured API access:
the model outputs{"name": "get weather",
"args":{"location": "Paris"}}, the ex-
ecutor invokes the function, and results are returned to
the model. GPT-4 achieves 95%+ accuracy on well-
documented APIs.
Code Interpreter[102] provides full programming en-
vironment: agents write Python code, execute in sandbox,
inspect output (including plots, files), and iterate. Exam-
ple: ”Analyze this CSV”→agent writespandascode,
generates visualization, interprets results. Enables com-
plex data analysis beyond API calls.
Toolformer[130] learns tool use via self-
supervision: during training, the model inserts tool
calls (<calculator>157 *23</calculator>),
executes them, and keeps calls that improve next-token
prediction. This teaches when (not just how) to use tools
without explicit labels.
9.7.6 Memory Architectures: Maintaining Long-
Term Context
Memory systems enable agents to accumulate knowledge,
recall relevant information, and maintain context beyond
fixed context windows. Table 9 categorizes memory ar-
chitectures.Retrieval-Augmented Generation (RAG)[81] stores
knowledge externally: given queryx, retrieve top-kdoc-
uments{d 1, . . . , d k}via dense retrieval (Contriever [72],
E5 [152]), then condition generation on[x;d 1;. . .;d k].
Enables access to massive corpora (Wikipedia, internal
documents) with 50-200ms latency.
Episodic Memory[110] stores interaction transcripts
with metadata (timestamp, importance scores): each
episode is embedded, and retrieval combines recency (ex-
ponential decay), importance (model-scored), and rele-
vance (embedding similarity). Generative Agents [110]
simulate 25 interactive characters maintaining consistent
personalities across 200+ interactions.
Parametric Memory[93] stores knowledge in model
weights: techniques like ROME [86] and MEMIT [87]
update specific neurons to inject new facts without full re-
training. Enables real-time knowledge updates with 0ms
retrieval but risks catastrophic forgetting.
9.8 Agent Communication Protocols: Stan-
dardizing Interfaces
As agentic systems proliferate, standardized protocols for
agent-tool communication and inter-agent coordination
are emerging to ensure interoperability and reduce inte-
gration overhead.
9.8.1 Model Context Protocol (MCP): LLM-Tool
Communication
Model Context Protocol[10] (MCP), introduced by An-
thropic (2024), standardizes how AI models interact with
external tools and data sources. MCP defines:
•Resources: Data sources the model can ac-
cess (files, databases, APIs) with URIs (e.g.,
file:///project/data.json).
•Prompts: Pre-defined prompt templates for com-
mon tasks, reusable across applications.
•Tools: Functions the model can invoke with JSON
schemas defining parameters and return types.
Architecture: MCP uses client-server model where:
•MCP Clients(AI applications like Claude Desktop,
IDEs) connect to multiple MCP servers.
56

•MCP Serversexpose tools/resources via
standardized interface. Example: filesys-
tem server providesread file(path),
list directory(path)tools.
•Transport Layer: JSON-RPC 2.0 over stdio (local
processes) or HTTP/SSE (remote servers).
Benefits: (1)Composability—single application ac-
cesses multiple tool servers without custom integrations,
(2)Security—servers control access scope (e.g., read-only
filesystem access), (3)Ecosystem Growth—developers
publish MCP servers for databases, APIs, specialized
tools, enabling plug-and-play functionality.
Example Workflow: User asks Claude to ”ana-
lyze sales data.” Claude’s MCP client queries avail-
able tools, findsdatabase query(sql)from MCP
database server, invokesSELECT *FROM sales
WHERE date > ’2024-01-01’, receives results,
and generates insights. All communication follows
MCP’s standardized JSON-RPC schema.
9.8.2 Agent-to-Agent (A2A) Protocols: Inter-Agent
Communication
While MCP handles agent-tool communication,Agent-
to-Agent (A2A)protocols address coordination between
autonomous agents. Key challenges include message se-
mantics, state synchronization, and coordination mecha-
nisms.
Communication Primitives:
•Message Passing: Agents exchange structured mes-
sages (requests, responses, notifications). Protocols
define message schemas (e.g., FIPA ACL [50] from
agent research) specifying sender, receiver, perfor-
mative (inform, request, propose), and content.
•Blackboard Architecture[29]: Shared memory
space where agents post/read information. Agent-
Verse [23] uses blackboards for dynamic task coor-
dination.
•Publish-Subscribe: Agents subscribe to topics;
publishers broadcast updates. Enables loose cou-
pling for large multi-agent systems.
Coordination Patterns:•Contract Net Protocol[140]: Agents negotiate task
assignments via bidding. Manager broadcasts task
→agents submit bids→manager awards contract
→agent executes task.
•Consensus Protocols: Multi-agent systems reach
agreement on shared state (e.g., which agent leads
subtask). Useful for collaborative planning without
central authority.
Emerging Standards: LangGraph [20] and Auto-
Gen [157] provide A2A frameworks supporting natu-
ral language messages with structured metadata (sender
role, task context, execution results). However, no uni-
versal A2A standard exists yet—current systems use
framework-specific protocols.
9.9 Agent Policies and Reward Structures:
Learning Optimal Behavior
Agentic systems must learn effective strategies for achiev-
ing goals. We formalize this throughpolicies(decision-
making rules) andreward structures(objectives to opti-
mize).
9.9.1 Policy Formulation: From Rules to Learned
Strategies
An agent’spolicyπmaps states to actions:π:S→A
(deterministic) orπ:S→∆(A)(stochastic distribution
over actions). For LLM agents:
πθ(at|st, ht) =P LLM(action text|state description,history)
(66)
whereθare model parameters,s tis current state (obser-
vations, tool outputs),h tis interaction history (previous
thoughts/actions), anda tis the next action (tool call, mes-
sage, reasoning step).
Policy Types:
•Prompting-Based Policies: Policy encoded in sys-
tem prompt (e.g., ”Use ReAct: alternate thoughts
and actions”). No learning—fixed strategy. Effec-
tive for well-specified tasks.
57

•Few-Shot Policies: Policy learned from examples in
context. Example: show 3 successful task trajecto-
ries→agent generalizes pattern. Limited by context
window.
•Fine-Tuned Policies: Policy learned via supervised
fine-tuning on agent trajectories. Example: collect
10K (state, action) pairs from expert agents, fine-
tune LLM:L=−Plogp θ(at|st, ht).
•RL-Optimized Policies: Policy learned via re-
inforcement learning. Proximal Policy Op-
timization (PPO) [131] optimizes:LPPO=
E[min(r t(θ)ˆAt,clip(r t(θ),1−ϵ,1 +ϵ) ˆAt)]where
rt(θ) =π θ(at|st)/πθold(at|st)is probability ratio,
ˆAtis advantage estimate.
Behavior Cloning vs. RL: Behavior cloning (super-
vised learning on expert trajectories) provides strong ini-
tialization but struggles with distribution shift (agent en-
counters states unseen in training data). RL enables on-
line learning—agent explores, observes outcomes, up-
dates policy—but requires well-defined reward signals
and significant compute.
9.9.2 Reward Modeling: Defining Success for Com-
plex Tasks
Specifying rewards for multi-step agent tasks is non-
trivial. Simply rewarding task completion ignores inter-
mediate quality (e.g., agent completes programming task
but generates unreadable code).
Reward Design Strategies:
•Sparse Terminal Rewards:r T= 1if goal
achieved,0otherwise. Simple but provides no learn-
ing signal during task. Requires many episodes to
learn.
•Dense Shaped Rewards: Provide intermediate
feedback:r t= ∆(progress toward goal). Example:
for web navigation, reward each successful subgoal
(login→search→add to cart). Faster learning but
requires task-specific engineering.
•Learned Reward Models: Train reward
modelR ϕ(s, a)from human preferences using
RLHF [109]. Collect comparisons: humans ratetrajectoriesτ 1> τ 2, train Bradley-Terry model:
p(τ1> τ 2) =σ(R ϕ(τ1)−R ϕ(τ2)). Agent then
optimizes learned reward via RL.
•Direct Preference Optimization (DPO)[119]:
Optimize policy directly from preferences
without explicit reward model:L DPO =
−E[logσ(βlogπθ(yw|x)
πref(yw|x)−βlogπθ(yl|x)
πref(yl|x))]where
yw, ylare preferred/dispreferred responses. Simpler
than RLHF; achieves comparable results.
Reward Hacking: Agents may exploit reward mis-
specification. Example: reward ”maximize code cover-
age”→agent writes meaningless tests achieving 100%
coverage but not testing functionality. Mitigation: com-
bine multiple reward signals (coverage + bug detection
rate + code quality), use learned reward models captur-
ing holistic preferences, implement human oversight for
high-stakes actions.
Constitutional AI[12]: Anthropic’s approach using
AI-generated self-critiques as reward signal. Agent gen-
erates response→critique model evaluates against princi-
ples (helpfulness, harmlessness, honesty)→agent revises
based on critique. Iterative refinement improves align-
ment without extensive human labeling.
9.10 Multi-Agent Coordination: Collabo-
rative Problem Solving
ReAct[166] pioneered practical agent planning by inter-
leaving reasoning with action: Thoughtt→Action t→
Observation t→Thoughtt+1. Example:Thought:
”Need population of Paris”→Action:Search[Paris
population]→Observation: ”2.1M residents”→
Thought: ”Can now compare to Tokyo.” This grounds rea-
soning in verifiable external state.
Reflexion[138] adds self-critique: after task failure,
the agent reflects on mistakes and generates improved
strategies. Trajectory: Attempt→Evaluate→Reflect
→Retry. On programming tasks, Reflexion improves
pass@1 by 30-50% through iterative debugging.
Tree-of-Thoughts (ToT)[165] enables systematic ex-
ploration: each thought is a partial solution, the model
evaluates quality and backtracks from dead ends. Com-
bines LLMs with search algorithms:V(s) =LLM eval(s)
scores intermediate states, guiding BFS/DFS exploration.
58

Table 11: Planning algorithm taxonomy for LLM agents, ranging from reactive (ReAct) to deliberative (Graph-of-
Thoughts) reasoning.
Algorithm Core Mechanism Strengths Limitations
ReAct[166] Interleaves reasoning traces with
action execution: Thought→Ac-
tion→Observation cycleGrounded in external
feedback; interpretable
reasoning chainsSequential execu-
tion; no backtrack-
ing
Reflexion[138] Learns from failures via self-
reflection: evaluates trajectory, gen-
erates feedback, retries with im-
proved strategyIterative improvement;
learns from mistakesRequires evaluation
signal; limited to
episodic tasks
Tree-of-
Thoughts[165]Explores solution tree with
BFS/DFS: generates multiple
thoughts, evaluates quality, back-
tracks when neededSystematic exploration;
handles complex plan-
ningHigh inference cost
(5-10×tokens); re-
quires value func-
tion
Graph-of-
Thoughts[16]Models thoughts as DAG enabling
parallel exploration and aggrega-
tionEfficient for paralleliz-
able tasks; combines di-
verse perspectivesComplex orchestra-
tion; requires task
decomposition ex-
pertise
Achieves state-of-the-art on Game of 24 (74%→100%)
but requires 5-10×more tokens.
Graph-of-Thoughts (GoT)[16] generalizes trees to
directed acyclic graphs, enabling parallel exploration and
aggregation. Example: generate 5 draft paragraphs con-
currently, then merge best elements into final text. Re-
duces latency for parallelizable tasks while maintaining
ToT-level quality.
9.10.1 Tool Use: Extending Models with External
Capabilities
Tool integration transforms LLMs from text generators
into versatile problem solvers accessing computation,
search, and specialized APIs. Table 8 categorizes tool-use
frameworks.
Function Calling[103] enables structured API access:
the model outputs{"name": "get weather",
"args":{"location": "Paris"}}, the ex-
ecutor invokes the function, and results are returned to
the model. GPT-4 achieves 95%+ accuracy on well-
documented APIs.
Code Interpreter[102] provides full programming en-
vironment: agents write Python code, execute in sandbox,
inspect output (including plots, files), and iterate. Exam-ple: ”Analyze this CSV”→agent writespandascode,
generates visualization, interprets results. Enables com-
plex data analysis beyond API calls.
Toolformer[130] learns tool use via self-
supervision: during training, the model inserts tool
calls (<calculator>157 *23</calculator>),
executes them, and keeps calls that improve next-token
prediction. This teaches when (not just how) to use tools
without explicit labels.
9.10.2 Memory Architectures: Maintaining Context
Across Interactions
Memory systems enable agents to accumulate knowledge,
recall relevant information, and maintain long-term con-
text beyond the fixed context window. Table 9 categorizes
memory architectures.
In-Context Memoryincludes all information directly
in the prompt. Modern models handle 32K-200K tokens
(GPT-4, Claude 3), enabling detailed conversation history
and documentation. Limitation: quadratic attention cost
(O(n2)) and fixed capacity.
Retrieval-Augmented Generation (RAG)[81] stores
knowledge externally: given queryx, retrieve top-kdoc-
uments{d 1, . . . , d k}via dense retrieval (contriever [72],
59

e5 [152]), then condition generation on[x;d 1;. . .;d k].
Enables access to massive corpora (Wikipedia, internal
documents) with 50-200ms retrieval latency.
Episodic Memory[110] stores interaction transcripts
with metadata (timestamp, importance scores): each
episode is embedded, and retrieval combines recency (ex-
ponential decay), importance (model-scored), and rele-
vance (embedding similarity). Generative Agents [110]
use episodic memory to simulate 25 interactive characters
maintaining consistent personalities across 200+ interac-
tions.
Parametric Memory[93] stores knowledge in model
weights: techniques like ROME [86] and MEMIT [87]
update specific neurons to inject new facts (”The Eiffel
Tower was painted green in 2024”) without full retraining.
Enables real-time knowledge updates with 0ms retrieval
but risks catastrophic forgetting.
9.11 Multi-Agent Coordination: Collabo-
rative Problem Solving
Multi-agent systems distribute tasks across specialized
agents with complementary capabilities, enabling com-
plex workflows beyond single-agent capacity. Key chal-
lenges include task decomposition, role assignment, com-
munication overhead, and coordination mechanisms.
9.11.1 Multi-Agent Frameworks and Coordination
Patterns
Table 12 categorizes coordination frameworks from con-
versational (AutoGen) to role-based (MetaGPT) to dy-
namic (AgentVerse) collaboration.
AutoGen[157] enables flexible agent conversations:
define agents with system prompts (e.g., ”You are a
Python expert”), tools (code executor), and termination
conditions. Example workflow: User describes task→
Assistant generates code→Executor runs code→As-
sistant debugs based on output→Repeat until success.
Supports human-in-the-loop for oversight.
MetaGPT[66] simulates software companies with
role-based workflows: (1) Product Manager writes PRD,
(2) Architect designs system (class diagrams, APIs), (3)
Engineer implements code, (4) QA tests and reports bugs.
Agents communicate via structured documents, achieving87% functional correctness on software generation bench-
marks (vs. 41% single-agent baseline).
ChatDev[113] implements waterfall development:
CEO defines objectives→CTO proposes tech stack→
Designer creates UI→Programmer codes→Tester val-
idates→Reviewer ensures quality. Chain-shaped com-
munication prevents chaos from full connectivity. Gen-
erates complete software (HTML/CSS/JS games, produc-
tivity tools) in ¡10 minutes.
AgentVerse[23] dynamically assembles teams: given
task description, a recruiter agent identifies required ex-
pertise (e.g., ”need Python expert, data scientist, visual-
ization specialist”) and instantiates agents with appropri-
ate prompts. Agents share state via blackboard architec-
ture. Outperforms fixed-role systems on diverse tasks re-
quiring variable expertise.
9.11.2 Emergent Behaviors and Coordination Chal-
lenges
Beneficial Emergence: Multi-agent systems exhibit
capabilities beyond individual agents. Example: In
MetaGPT, architect agent proposes modular design (not
explicitly prompted), enabling engineer agent to imple-
ment components independently, then integration agent
combines them. Modularity emerges from role special-
ization.
Coordination Overhead: Communication costs scale
quadratically with agent count. Solutions: (1) hierarchical
organization (manager agent coordinates sub-teams), (2)
sparse communication topologies (chain/tree structures),
(3) asynchronous message passing (agents don’t wait for
responses).
Conflicting Objectives: Agents may pursue incompat-
ible goals. Example: In software development, engineer
optimizes for feature richness while QA agent optimizes
for bug-free code—tension requires negotiation or meta-
agent arbitration.
Consensus Mechanisms: When agents must agree
on decisions (e.g., which design to implement), systems
use voting (majority/plurality), debate (agents argue posi-
tions, final decision via critique), or hierarchical authority
(designated agent makes final call).
60

Table 12: Multi-agent coordination frameworks: from conversational (AutoGen) to role-based (MetaGPT) to dynamic
(AgentVerse) collaboration.
Framework Coordination Mechanism Agent Roles Communication
AutoGen[157] Conversational agents with customiz-
able roles; human-in-the-loop; code
executionUser proxy, assis-
tant, executorNatural language
messages
MetaGPT[66] Simulates software company: struc-
tured workflow (requirements→de-
sign→code→test)PM, architect, engi-
neer, QAStructured documents
(PRD, design docs)
ChatDev[113] Chain-shaped collaboration: CEO→
CTO→programmer→reviewer; wa-
terfall development7 roles (CEO,
CTO, designer,
programmer, tester,
reviewer, art de-
signer)Sequential handoffs
with code/design
artifacts
AgentVerse[23] Dynamic task decomposition: re-
cruiter agent assembles team based on
task requirementsDynamic role as-
signmentBlackboard architec-
ture (shared memory)
9.12 Evaluation and Benchmarking for
Agentic Systems
Evaluating agentic capabilities requires moving beyond
static benchmarks to interactive, multi-step tasks with ver-
ifiable outcomes. Key evaluation frameworks:
WebArena[171]: Agents navigate realistic web envi-
ronments (e-commerce, forums, CMS) to complete tasks
(”Find products under $50 and add to cart”). Evaluates
tool use, planning, and grounding. GPT-4 achieves 14.4%
success rate, highlighting remaining challenges.
SWE-bench[77]: Agents resolve real GitHub issues in
popular repositories. Requires understanding codebases,
writing patches, passing tests. State-of-the-art: 13.8% is-
sues resolved (vs. 0.8% for baseline models), demonstrat-
ing rapid progress in code agents.
GAIA[88]: General AI Assistants benchmark with
multi-modal, multi-step questions requiring web search,
code execution, file processing. Only 15% solvable
by current systems, providing headroom for long-term
progress.
AgentBench[85]: 8 environments spanning web navi-
gation, database querying, game playing, household tasks
(ALFWorld). Unified evaluation protocol: success rate
on task completion. GPT-4 achieves 52% average success
(vs. 28% for GPT-3.5), validating capability differences.Key Insight: Agentic benchmarks reveal current
limitations—even frontier models solve ¡20% of com-
plex real-world tasks—highlighting the gap between text
generation capabilities and embodied problem-solving.
Progress requires improved planning (handling 10+ step
horizons), robust error recovery (agents currently fail
catastrophically on errors), and efficient exploration (re-
ducing trial-and-error overhead).
9.13 Safety, Oversight, and Risk Mitigation
Agentic systems with tool access and autonomous ex-
ecution raise critical safety concerns: unintended ac-
tions (e.g., irreversible API calls), exploration of harmful
strategies, and accumulation of errors. We identify three
key safety dimensions:
9.13.1 Human Oversight Mechanisms
Preventing unintended autonomy while maintaining effi-
ciency:
Action Confirmation: Require human approval for
high-stakes actions (financial transactions, data deletion,
external communications). AutoGen implements config-
urable approval: critical actions pause for human confir-
61

mation, while safe actions (read-only queries) proceed au-
tonomously.
Interpretable Reasoning Traces: Log all thoughts,
actions, and observations for post-hoc auditing. Re-
Act’s explicit reasoning chains enable humans to identify
flawed logic: ”Thought: I’ll delete all files to free space”
can be caught before execution.
Capability Restrictions: Limit tool access based on
task context. Example: data analysis agents access
pandasbut notos.system(), preventing inadvertent
system modifications. GPT-4 Code Interpreter runs in iso-
lated sandbox (no network, ephemeral filesystem).
9.13.2 Red-Teaming and Adversarial Testing
Proactive identification of failure modes:
Goal Misalignment Probes: Test whether agents pur-
sue unintended interpretations. Example: ”Maximize user
engagement”→agent generates addictive/misleading
content. Anthropic’s Constitutional AI [12] uses auto-
mated red-teaming to identify harmful strategies.
Tool Misuse Detection: Monitor for unexpected API
usage patterns. Example: agent bypasses rate limits by
spawning multiple sessions. Anomaly detection flags de-
viations from expected tool-use distributions.
Multi-Agent Collusion: Test whether independent
agents cooperate to circumvent restrictions. Example:
Agent A gathers restricted information, Agent B executes
actions based on that information. Requires system-level
monitoring beyond individual agent oversight.
9.13.3 Graceful Failure and Containment
Ensuring errors don’t cascade:
Reversible Actions: Design APIs with undo mecha-
nisms. Example: database agents use transactions (BE-
GIN/COMMIT/ROLLBACK), enabling rollback on er-
rors. File operations create backups before modifications.
Timeouts and Circuit Breakers: Halt execution af-
ter excessive retries or resource consumption. Example:
agent stuck in exploration loop exhausts 1000-action bud-
get, triggering forced termination with informative error.
Sandboxing and Virtualization: Execute actions in
isolated environments (containers, VMs) preventing unin-
tended impact on production systems. Explosion radius:
contain failures to sandbox, no lateral movement.Case Study: OpenAI Code Interpreter implements
multi-layer safety: (1) sandboxed Python environment (no
network, limited filesystem), (2) execution timeout (120
seconds per cell), (3) explicit user confirmation for file
downloads, and (4) automatic session termination after 60
minutes idle. These constraints reduce risk while enabling
broad utility for data analysis and visualization.
Open Challenge: Current oversight mechanisms rely
on human monitoring or rule-based guardrails, both brit-
tle at scale. Scaling to fully autonomous agents (e.g.,
personal assistants with 24/7 access to email, calendar,
finances) requireslearned safety modelsthat predict ac-
tion consequences anduncertainty quantificationto re-
quest help when confidence is low. Recent work on Con-
stitutional AI [12] and debate-based oversight [71] offers
promising directions but remains far from deployment-
ready for high-stakes autonomy.
10 Key Findings, Open Challenges,
and Future Directions
This survey has traced the remarkable evolution of arti-
ficial intelligence from the foundational Transformer ar-
chitecture (2017) to the sophisticated reasoning-capable
systems of 2025. Through comprehensive analysis of
over 50 major models, detailed examination of training
methodologies, and rigorous benchmarking across mul-
tiple dimensions, we have documented three fundamental
paradigm shifts that define the current state and future tra-
jectory of AI research.
10.1 Key Findings and Paradigm Shifts
From Pre-Training to Post-Training DominanceOur
analysis reveals that the locus of capability development
has decisively shifted from pre-training scale to post-
training optimization. While GPT-3 (2020) demonstrated
that pre-training alone could achieve 70% MMLU perfor-
mance, modern systems attribute 70-90% of their capa-
bilities to reinforcement learning techniques. DeepSeek-
R1’s achievement of 79.8% on MATH and 79.2%
on AIME through pure RL—without supervised fine-
tuning—validates that verifiable rewards alone can induce
sophisticated reasoning. This shift has profound impli-
cations:data quality matters more than quantity,human
62

feedback becomes the bottleneck, andautomated feedback
mechanisms offer scaling paths beyond human judgment.
Test-Time Compute as a New Scaling Dimension
The emergence of o1, DeepSeek-R1, and reasoning-
specialized models establishes test-time compute as a vi-
able alternative to pre-training scale. Our documentation
shows that performance∝log(inference FLOPs), where
1 hour of extended reasoning achieves gains equivalent
to106×training compute. This represents a fundamen-
tal rebalancing: rather than concentrating all compute in
pre-training, systems can strategically allocate resources
between training and inference based on task complex-
ity. The economic implications are significant—test-time
scaling enables smaller, cheaper base models to achieve
GPT-4-level performance on reasoning tasks through ex-
tended thinking time.
The Efficiency RevolutionInnovations documented
in this survey—Multi-head Latent Attention (8×KV
cache compression), mixture-of-experts with fine-grained
routing (18×parameter efficiency in DeepSeek-V3),
FlashAttention-2 (2.5×speedup), GQA (4-8×KV reduc-
tion), and optimized inference stacks—have collectively
achieved a 100×cost reduction in 2 years. GPT-4-level
performance now costs<$0.30/M tokens versus $30+/M
tokens in 2023. This democratization enables broader ac-
cess, more extensive experimentation, and deployment in
cost-sensitive applications previously inaccessible to AI.
10.2 The Scaling Wall and Post-Scaling Era
A central theme of this survey is the scaling wall con-
fronting continued model development. The convergence
of three crises—data exhaustion (9-27T tokens depleted
by 2026-2028), exponential cost growth ($3M to $300M+
in 5 years), and unsustainable energy consumption (22×
increase from GPT-3 to GPT-4)—fundamentally chal-
lenges naive scaling. As Coveney et al. [30] and
Hooker [67] document, the simple formula of ”scale
model size and training data” has become inadequate.
Yet this constraint has catalyzed innovation precisely
along the three alternative paradigms we have analyzed:
(1) test-time compute scaling demonstrates that reason-
ing can be ”purchased” at inference time, (2) sparse ar-chitectures like MoE prove that parameter efficiency can
match dense model quality, and (3) architectural innova-
tions (linear attention, state space models, sparse patterns)
break complexity barriers. The transition from ”scal-
ing era” to ”efficiency era” mirrors computing’s histor-
ical pivot from frequency scaling to multicore architec-
tures and specialized accelerators when Moore’s Law ap-
proached limits.
10.3 Open-Source Democratization and
Global Competition
The competitive landscape has fundamentally trans-
formed. Open-source models now match or exceed
proprietary systems: Llama 3 (405B) achieves 88.6%
MMLU versus GPT-4’s 86.4%, Qwen-3 demonstrates
reasoning capabilities competitive with Claude 3.5, and
DeepSeek-V3 delivers GPT-4-level performance at 18×
parameter efficiency. This democratization enables:
•Research acceleration: Academic institutions can
now train and experiment with frontier-class models
using accessible compute
•Regional innovation: Organizations across China
(Alibaba, ByteDance, Moonshot, Zhipu), Europe,
and other regions contribute diverse architectural ap-
proaches
•Specialized applications: Domain-specific models
fine-tuned for healthcare, code generation, scientific
reasoning, and multilingual tasks
•Safety research: Open weights enable independent
analysis of model behavior, bias, and failure modes
The emergence of globally distributed AI capability
breaks the concentration of progress in a handful of US-
based labs, introducing beneficial competition, diverse
perspectives, and accelerated innovation cycles.
10.4 Critical Open Challenges
Despite remarkable progress, fundamental challenges re-
main:
63

Verifiable Reasoning at ScaleWhile DeepSeek-R1
demonstrates that pure RL can induce reasoning through
verifiable rewards (MATH problems have ground truth),
extending this to open-ended domains where ground truth
is ambiguous or subjective remains unsolved. The field
lacks robust frameworks for: (1) evaluating reasoning
chains beyond final answers, (2) detecting subtle logical
errors in multi-step derivations, (3) balancing exploration
of reasoning strategies with exploitation of known pat-
terns, and (4) scaling human oversight of complex reason-
ing processes. Constitutional AI [12] and debate-based
methods offer promising directions, but achieving ”pro-
cess supervision at scale” remains an open problem.
Efficient Test-Time Compute AllocationTest-time
scaling introduces new optimization challenges:whento
allocate extended compute (not all queries require deep
thinking),how muchcompute to allocate (dynamic bud-
geting based on query complexity), andhowto search ef-
ficiently (balancing breadth-first exploration with depth-
first refinement). Current systems use simple fixed-budget
approaches, but optimal policies likely require learned
meta-strategies that predict query difficulty and adaptively
allocate resources. This connects to classical problems
in heuristic search but at unprecedented scale and with
learned rather than handcrafted heuristics.
Multimodal Integration Beyond ConcatenationCur-
rent multimodal models largely concatenate visual and
text tokens or use shallow cross-attention. True multi-
modal understanding requires: (1) aligned representations
where semantically equivalent concepts across modalities
have similar encodings, (2) cross-modal reasoning where
information from one modality guides processing in an-
other, (3) temporal synchronization for video and audio,
and (4) efficient training from limited paired data. Models
like Gemini 1.5 and GPT-4o demonstrate impressive mul-
timodal capabilities, but systematic understanding of what
constitutes good multimodal architecture remains elusive.
Agent Reliability and SafetyAs LLMs evolve into au-
tonomous agents using tools, accessing external systems,
and making consequential decisions, reliability becomes
critical. Challenges include: (1) hallucination mitigationwhen agents generate and execute code, (2) failure re-
covery when tool calls fail or return unexpected results,
(3) constraint satisfaction ensuring agents respect secu-
rity boundaries and ethical guidelines, (4) compositional
generalization to novel task combinations, and (5) inter-
pretability of agent decision-making for debugging and
oversight. The transition from ”helpful assistant” to ”au-
tonomous agent” requires solving these problems at scale.
Economic Sustainability and Environmental Impact
While efficiency innovations have dramatically reduced
per-token costs, aggregate energy consumption contin-
ues growing as usage expands. Training DeepSeek-V3
(the most efficient frontier model) still required $110M
and 2,279 MWh—equivalent to 200+ households’ annual
electricity. Sustainable AI requires: (1) continued al-
gorithmic efficiency gains beyond current 100×cost re-
ductions, (2) hardware co-design optimizing for trans-
former/SSM operations, (3) dynamic resource allocation
to avoid over-provisioning, (4) renewable energy integra-
tion for large training runs, and (5) economic models that
internalize environmental costs. The field must balance
capability progress with planetary constraints.
From Human Preference to Outcome Optimization
Current RLHF systems optimize for human prefer-
ence—what responses humanslike—which may diverge
from what is objectivelycorrectoruseful. For tasks with
verifiable outcomes (mathematics, code execution, sci-
entific reasoning), the field is transitioning to outcome-
based training. However, most real-world applications
lack clear ground truth. Open questions include: (1) de-
signing reward functions for subjective domains (creative
writing, strategic advice), (2) balancing short-term user
satisfaction with long-term value, (3) aggregating prefer-
ences across diverse user populations, (4) handling distri-
bution shift as model capabilities evolve, and (5) detect-
ing and correcting reward hacking where models exploit
loopholes in feedback mechanisms.
10.5 Future Research Directions
Building on the foundations established in this survey, we
identify high-priority research directions:
64

1. Hybrid Architectures Combining Transformers,
SSMs, and Graph NetworksNo single architecture
dominates all dimensions. Transformers excel at in-
context learning but suffer quadratic complexity; SSMs
(Mamba) achieve linear complexity but struggle with cer-
tain reasoning patterns; graph networks capture structured
relationships but require domain-specific design. Future
systems may combine these paradigms: using SSMs for
efficient sequence encoding, Transformers for critical rea-
soning steps requiring global context, and graph networks
for explicit relational reasoning. Learned routing mech-
anisms could dynamically select architectures based on
input characteristics.
2. Continual Learning and Efficient Model Updates
Current models are static—trained once on a fixed dataset,
then deployed. Real-world deployment requires: (1) in-
corporating new information without full retraining (con-
tinual learning), (2) correcting errors and updating knowl-
edge (model editing), (3) personalizing to user prefer-
ences over time, and (4) forgetting sensitive informa-
tion on request. Techniques like parameter-efficient fine-
tuning (LoRA, QLoRA), elastic weight consolidation,
and compositional model architectures offer partial solu-
tions, but systematic frameworks for lifelong learning at
scale remain underdeveloped.
3. Interpretability and Mechanistic Understanding
Despite impressive capabilities, our mechanistic under-
standing of how large models work remains limited. Pri-
ority questions include: (1) what algorithms do transform-
ers implement (how do they perform in-context learn-
ing, arithmetic, logical reasoning?), (2) what represen-
tations emerge in hidden states (are there interpretable
”concepts”?), (3) how does reasoning emerge from scale
and training (phase transitions, critical thresholds), (4)
what causes failure modes (hallucination, biased outputs,
capability gaps), and (5) how can we verify model be-
havior (proofs of reasoning correctness)? Progress re-
quires both empirical investigation (probing, intervention
experiments) and theoretical analysis (mechanistic inter-
pretability, circuits perspective).
4. Efficient Reasoning Search AlgorithmsTest-time
compute scaling raises algorithmic questions: what searchstrategies optimize reasoning under compute budgets?
Current approaches use beam search or best-of-N sam-
pling, but these are generic. Domain-specific algorithms
may outperform: mathematical theorem proving might
benefit from proof search guided by learned heuristics,
code generation from program synthesis with type-guided
search, scientific reasoning from causal inference algo-
rithms. Integrating classical AI search methods (A*,
IDA*, Monte Carlo Tree Search) with learned models of-
fers rich opportunities.
5. Synthetic Data and Self-Improvement LoopsAs
high-quality human data depletes, synthetic data genera-
tion becomes critical. Key challenges: (1) ensuring syn-
thetic data quality (avoiding model collapse from train-
ing on own outputs), (2) designing curricula that target
capability gaps, (3) using verification to filter generated
data (proof checkers for math, compilers for code, sim-
ulators for robotics), (4) bootstrapping from weak mod-
els to strong models, and (5) understanding theoretical
limits of self-improvement. AlphaGeometry’s synthetic
theorem proving and DeepSeek-R1’s RL-only training
demonstrate feasibility, but systematic frameworks re-
main open.
6. Multimodal Foundation Models for Robotics and
Embodied AIText-based models demonstrate impres-
sive reasoning, but real-world deployment requires phys-
ical grounding. Embodied AI research must address: (1)
learning from limited real-world interaction data (sim-
to-real transfer, foundation models pre-trained on inter-
net data then adapted to robotics), (2) real-time inference
under compute constraints (embedded deployment), (3)
safety and robustness (physical systems have irreversible
consequences), (4) common sense physics (understand-
ing object interactions, stability, causality), and (5) long-
horizon task planning (combining high-level reasoning
with low-level control). Models like RT-2 and PaLM-
E represent early steps, but general-purpose embodied
agents remain distant.
11 Conclusion
The evolution from GPT-2 (2019, 1.5B parameters)
to DeepSeek-V3 (2025, 671B parameters, 37B ac-
65

tive) represents unprecedented technological accelera-
tion, achieving three fundamental breakthroughs: (1)
post-training dominance—70-90% of capabilities now
derive from RLHF/RL rather than pre-training scale,
(2)test-time compute scaling—inference-time reason-
ing achieves gains equivalent to106×training compute,
and (3)100×efficiency gains—cost per token reduced
from $30+ to $0.30 while maintaining GPT-4-level qual-
ity.
Three converging crises define the scaling wall: data
exhaustion (9-27T tokens depleted by 2026-2028), expo-
nential cost growth ($3M to $300M+ in 5 years), and
22×energy increase (GPT-3 to GPT-4). Yet these con-
straints have catalyzed innovation: sparse architectures
(MoE achieving 18×efficiency), sub-quadratic attention
mechanisms (linear attention, state space models), and ar-
chitectural creativity prove that continued progress need
not rely on brute-force scaling.
Open-source democratization has fundamentally trans-
formed the competitive landscape. Llama 3 (88.6%
MMLU) surpasses GPT-4 (86.4%), enabling academic re-
search, regional innovation across China/Europe, and in-
dependent safety analysis. This global distribution of AI
capability introduces beneficial competition and acceler-
ates innovation cycles.
Critical challenges remain: verifiable reasoning at scale
beyond ground-truth domains, efficient test-time com-
pute allocation, true multimodal integration, agent relia-
bility and safety, environmental sustainability, and tran-
sitioning from human preference to outcome optimiza-
tion. The field’s trajectory depends on solving these prob-
lems while maintaining the remarkable pace of capability
growth documented in this survey.
This work provides comprehensive technical docu-
mentation—rigorous mathematical formulations (RLHF,
PPO, DPO, GRPO), architectural analysis (FlashAtten-
tion, GQA, MoE, Mamba), and benchmarking across 9
metrics spanning 30+ models—serving as both practi-
tioner reference and research foundation. As the field
transitions from the scaling era to the efficiency era, the
ingenuity demonstrated in navigating current constraints,
the diversity of global contributions, and rapid democrati-
zation suggest the post-scaling era may prove even more
innovative than the scaling era itself.References
[1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao,
Harkirat Behl, et al. Phi-4 technical report.arXiv
preprint arXiv:2412.08905, 2024.
[2] Marah Abdin, Sam Ade Jacobs, Ammar Ah-
mad Awan, Jyoti Aneja, Ahmed Awadallah, Hany
Awadalla, Nguyen Bach, Amit Bahree, Arash
Bakhtiari, Harkirat Behl, et al. Phi-3 technical
report: A highly capable language model locally
on your phone.arXiv preprint arXiv:2404.14219,
2024.
[3] Armen Aghajanyan, Sonal Gupta, and Luke Zettle-
moyer. Intrinsic dimensionality explains the effec-
tiveness of language model fine-tuning.Proceed-
ings of the 59th Annual Meeting of the Associa-
tion for Computational Linguistics, pages 7319–
7328, 2021. Foundation for understanding low-
rank adaptation methods.
[4] AI2. Olmo 2: Post-norm architecture and training
stability.arXiv preprint, 2025.
[5] AI2. Olmo 3 think: Open reasoning model with
full transparency.arXiv preprint, 2025.
[6] Joshua Ainslie, James Lee-Thorp, Michiel de Jong,
Yury Zemlyanskiy, Federico Lebr ´on, and Sumit
Sanghai. Gqa: Training generalized multi-query
transformer models from multi-head checkpoints.
arXiv preprint arXiv:2305.13245, 2023.
[7] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc,
Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katie Millican, Malcolm
Reynolds, et al. Flamingo: a visual language model
for few-shot learning.Advances in Neural Informa-
tion Processing Systems, 35:23716–23736, 2022.
[8] Alibaba Cloud. Qwen 3: Advancing open-source
language models.arXiv preprint, 2025.
[9] Alibaba Cloud. Qwen3-next: Hybrid architecture
with gated deltanet.arXiv preprint, 2025.
66

[10] Anthropic. Model context protocol (MCP): Stan-
dardizing AI-tool communication.https://
modelcontextprotocol.io, 2024. Protocol
for standardized communication between AI mod-
els and external tools/data sources using JSON-
RPC.
[11] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua
Bengio. Neural machine translation by jointly
learning to align and translate. InInternational
Conference on Learning Representations, 2015.
[12] Yuntao Bai, Saurav Kadavath, Sandipan Kundu,
Amanda Askell, Jackson Kernion, Andy Jones,
Anna Chen, Anna Goldie, Azalia Mirhoseini,
Cameron McKinnon, et al. Constitutional ai:
Harmlessness from ai feedback.arXiv preprint
arXiv:2212.08073, 2022.
[13] Iz Beltagy, Matthew E Peters, and Arman Co-
han. Longformer: The long-document transformer.
arXiv preprint arXiv:2004.05150, 2020.
[14] Yoshua Bengio, Nicholas L ´eonard, and Aaron
Courville. Estimating or propagating gradients
through stochastic neurons for conditional compu-
tation. InarXiv preprint arXiv:1308.3432, 2013.
Straight-through estimator for gradient approxima-
tion.
[15] Tamay Besiroglu, Jaime Sevilla, Lennart Heim,
Marius Hobbhahn, Pablo Villalobos, and David
Owen. The rising costs of training frontier ai mod-
els.arXiv preprint arXiv:2405.21015, 2024.
[16] Maciej Besta, Nils Blach, Ales Kubicek, Robert
Gerstenberger, Lukas Gianinazzi, Joanna Gajda,
Tomasz Lehmann, Michal Podstawski, Hubert
Niewiadomski, Piotr Nyczyk, et al. Graph
of thoughts: Solving elaborate problems
with large language models.arXiv preprint
arXiv:2308.09687, 2023. Extends tree-of-thoughts
to DAG structure enabling parallel exploration and
aggregation.
[17] Ralph Allan Bradley and Milton E Terry. Rank
analysis of incomplete block designs: I. the methodof paired comparisons.Biometrika, 39(3/4):324–
345, 1952. Bradley-Terry model for pairwise pref-
erence modeling.
[18] Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sas-
try, Amanda Askell, et al. Language models are
few-shot learners.Advances in Neural Information
Processing Systems, 33:1877–1901, 2020.
[19] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu
Peng, and Tri Dao. Medusa: Simple llm infer-
ence acceleration framework with multiple decod-
ing heads.arXiv preprint arXiv:2401.10774, 2024.
[20] Harrison Chase. Langchain: Building applications
with llms through composability.GitHub reposi-
tory, 2023.
[21] Charlie Chen, Sebastian Borgeaud, Geoffrey Irv-
ing, Jean-Baptiste Lespiau, Laurent Sifre, and John
Jumper. Accelerating large language model de-
coding with speculative sampling.arXiv preprint
arXiv:2302.01318, 2023.
[22] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming
Yuan, Henrique Ponde de Oliveira Pinto, Jared Ka-
plan, Harri Edwards, Yuri Burda, Nicholas Joseph,
Greg Brockman, et al. Evaluating large lan-
guage models trained on code.arXiv preprint
arXiv:2107.03374, 2021.
[23] Weize Chen, Yusheng Su, Jingwei Zuo, Cheng
Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan,
Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agent-
Verse: Facilitating multi-agent collaboration and
exploring emergent behaviors.arXiv preprint
arXiv:2308.10848, 2023. Dynamic team assembly
with blackboard architecture for variable expertise
requirements.
[24] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng,
Anastasios Nikolas Angelopoulos, Tianle Li,
Dacheng Li, Hao Zhang, Banghua Zhu, Michael
Jordan, Joseph E Gonzalez, et al. Chatbot arena:
An open platform for evaluating llms by hu-
man preference.arXiv preprint arXiv:2403.04132,
2024.
67

[25] Rewon Child, Scott Gray, Alec Radford, and Ilya
Sutskever. Generating long sequences with sparse
transformers. InarXiv preprint arXiv:1904.10509,
2019. Sparse attention patterns for efficient long-
sequence modeling.
[26] Paul Christiano, Buck Shlegeris, and Dario
Amodei. Supervising strong learners by
amplifying weak experts.arXiv preprint
arXiv:1810.08575, 2018. Scalable oversight
through iterated amplification and distillation.
[27] Karl Cobbe, Vineet Kosaraju, Mohammad Bavar-
ian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton,
Reiichiro Nakano, et al. Training verifiers to
solve math word problems.arXiv preprint
arXiv:2110.14168, 2021.
[28] Cognition Labs. Introducing devin: The
first ai software engineer.https:
//www.cognition-labs.com/
introducing-devin, 2024. Autonomous AI
coding agent with end-to-end software develop-
ment capabilities.
[29] Daniel D Corkill. Blackboard systems.AI ex-
pert, 6(9):40–47, 1991. Blackboard architecture:
shared memory space for multi-agent coordination
and problem-solving.
[30] Peter V . Coveney and Sauro Succi. The wall con-
fronting large language models.arXiv preprint
arXiv:2507.19703, 2025. Demonstrates that scal-
ing laws severely limit LLMs’ ability to improve
prediction uncertainty and reliability.
[31] Tri Dao. Flashattention-2: Faster attention with
better parallelism and work partitioning.arXiv
preprint arXiv:2307.08691, 2023.
[32] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and
Christopher R ´e. Flashattention: Fast and memory-
efficient exact attention with io-awareness.Ad-
vances in Neural Information Processing Systems,
35:16344–16359, 2022.
[33] DeepSeek-AI. Deepseek-coder: When the
large language model meets programming–therise of code intelligence.arXiv preprint
arXiv:2401.14196, 2024.
[34] DeepSeek-AI. Deepseek llm: Scaling open-source
language models with longtermism.arXiv preprint
arXiv:2401.02954, 2024.
[35] DeepSeek-AI. Deepseek-v3 technical report.arXiv
preprint arXiv:2412.19437, 2024.
[36] DeepSeek-AI. Deepseekmath: Pushing the limits
of mathematical reasoning in open language mod-
els.arXiv preprint arXiv:2402.03300, 2024.
[37] DeepSeek-AI. Deepseekmoe: Towards ultimate
expert specialization in mixture-of-experts lan-
guage models.arXiv preprint arXiv:2401.06066,
2024.
[38] DeepSeek-AI. Deepseekmoe: Towards ultimate
expert specialization in mixture-of-experts lan-
guage models.arXiv preprint arXiv:2401.06066,
2024.
[39] DeepSeek-AI. Deepseek-r1: Incentivizing reason-
ing capability in llms via reinforcement learning.
arXiv preprint arXiv:2501.12948, 2025.
[40] DeepSeek-AI. Deepseek-v3 technical report.arXiv
preprint arXiv:2412.19437, 2025.
[41] Mostafa Dehghani, Josip Djolonga, Basil Mustafa,
Piotr Padlewski, Jonathan Heek, Justin Gilmer,
Andreas Steiner, Mathilde Caron, Robert Geirhos,
Ibrahim Alabdulmohsin, et al. Scaling vision trans-
formers to 22 billion parameters.arXiv preprint
arXiv:2302.05442, 2023.
[42] Tim Dettmers, Artidoro Pagnoni, Ari Holtz-
man, and Luke Zettlemoyer. Qlora: Efficient
finetuning of quantized llms.arXiv preprint
arXiv:2305.14314, 2023.
[43] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman,
and Luke Zettlemoyer. QLoRA: Efficient finetun-
ing of quantized LLMs.Advances in Neural In-
formation Processing Systems, 36, 2024. NeurIPS
2023 proceedings published in 2024. 4-bit quanti-
zation with backpropagation through frozen quan-
tized weights.
68

[44] Jacob Devlin, Ming-Wei Chang, Kenton Lee,
and Kristina Toutanova. Bert: Pre-training of
deep bidirectional transformers for language un-
derstanding.arXiv preprint arXiv:1810.04805,
2019.
[45] Danny Driess, Fei Xia, Mehdi SM Sajjadi,
Corey Lynch, Aakanksha Chowdhery, Brian Ichter,
Ayzaan Wahid, Jonathan Tompson, Quan Vuong,
Tianhe Yu, et al. Palm-e: An embodied multi-
modal language model.International Conference
on Machine Learning (ICML), pages 8469–8488,
2023. Embodied multimodal model integrating vi-
sion and language for robotics.
[46] Abhimanyu Dubey, Abhinav Jauhri, Abhinav
Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten,
Amy Yang, Angela Fan, et al. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783, 2024.
[47] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi
Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos
Guestrin, Percy Liang, and Tatsunori B Hashimoto.
Alpacafarm: A simulation framework for methods
that learn from human feedback.arXiv preprint
arXiv:2305.14387, 2023.
[48] Elicit. Elicit: The ai research assistant.https:
//elicit.org, 2024. AI assistant for literature
review and research synthesis.
[49] Epoch AI. Can ai scaling continue through 2030?
Epoch AI Research, 2024.
[50] FIPA. FIPA ACL message structure specification.
Foundation for Intelligent Physical Agents, 2002.
FIPA Agent Communication Language: standard-
ized agent message protocols with performatives.
[51] Elias Frantar, Saleh Ashkboos, Torsten Hoefler,
and Dan Alistarh. Gptq: Accurate post-training
quantization for generative pre-trained transform-
ers.arXiv preprint arXiv:2210.17323, 2023.
[52] Gemini Team. Gemini: A family of highly
capable multimodal models.arXiv preprint
arXiv:2312.11805, 2023.[53] Charles Goddard, Shamane Siriwardhana, Malikeh
Ehghaghi, et al. Arcee’s MergeKit: A toolkit for
merging large language models.arXiv preprint
arXiv:2403.13257, 2024.
[54] Google DeepMind. Gemma 2: Improving open
language models at a practical size.arXiv preprint,
2024.
[55] Google DeepMind. Gemma 3: Aggressive sliding
window attention with 5:1 ratio.arXiv preprint,
2025.
[56] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Ak-
shita Bhagia, Rodney Kinney, Oyvind Tafjord,
Ananya Harsh Jha, Hamish Ivison, Ian Magnus-
son, Yizhong Wang, et al. Olmo: Accelerating
the science of language models.arXiv preprint
arXiv:2402.00838, 2024.
[57] Albert Gu and Tri Dao. Mamba: Linear-time se-
quence modeling with selective state spaces.arXiv
preprint arXiv:2312.00752, 2023.
[58] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio
C´esar Teodoro Mendes, Allie Del Giorno,
Sivakanth Gopi, Mojan Javaheripi, Piero Kauff-
mann, Gustavo de Rosa, Olli Saarikivi, et al.
Textbooks are all you need.arXiv preprint
arXiv:2306.11644, 2023. Phi-1 model paper.
[59] David Ha and J ¨urgen Schmidhuber. World models.
arXiv preprint arXiv:1803.10122, 2018.
[60] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba,
and Timothy Lillicrap. Mastering diverse do-
mains through world models.arXiv preprint
arXiv:2301.04104, 2023.
[61] Adi Haviv, Jonathan Berant, and Amir Glober-
son. Understanding masked self-attention as
implicit positional encoding.arXiv preprint
arXiv:2310.04393, 2023.
[62] Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob
Steinhardt. Measuring massive multitask language
understanding.arXiv preprint arXiv:2009.03300,
2021.
69

[63] Jonathan Ho, Ajay Jain, and Pieter Abbeel. De-
noising diffusion probabilistic models.Advances in
Neural Information Processing Systems, 33:6840–
6851, 2020.
[64] Jordan Hoffmann, Sebastian Borgeaud, Arthur
Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hen-
dricks, Johannes Welbl, Aidan Clark, et al. Train-
ing compute-optimal large language models.arXiv
preprint arXiv:2203.15556, 2022.
[65] Jiwoo Hong, Noah Lee, and James Thorne. ORPO:
Monolithic preference optimization without refer-
ence model.arXiv preprint arXiv:2403.07691,
2024.
[66] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xi-
awu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin
Wang, Zili Wang, Steven Ka Shing Yau, Zijuan
Lin, et al. Metagpt: Meta programming for multi-
agent collaborative framework.arXiv preprint
arXiv:2308.00352, 2023.
[67] Sara Hooker. On the slow death of scaling.SSRN
Electronic Journal, 2025. Available at SSRN:
https://ssrn.com/abstract=5877662.
[68] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzeb-
ski, Bruna Morrone, Quentin De Laroussilhe, An-
drea Gesmundo, Mona Attariyan, and Sylvain
Gelly. Parameter-efficient transfer learning for nlp.
InInternational Conference on Machine Learning,
pages 2790–2799. PMLR, 2019.
[69] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. Lora: Low-rank adapta-
tion of large language models.arXiv preprint
arXiv:2106.09685, 2021.
[70] Wenlong Huang, Chen Wang, Ruohan Zhang, Yun-
zhu Li, Jiajun Wu, and Li Fei-Fei. V oxposer:
Composable 3d value maps for robotic manip-
ulation with language models.arXiv preprint
arXiv:2307.05973, 2023. LLM-based framework
for robot manipulation via composable 3D affor-
dances.[71] Geoffrey Irving, Paul Christiano, and Dario
Amodei. AI safety via debate.arXiv preprint
arXiv:1805.00899, 2018. Proposes debate-based
oversight for AI safety through adversarial interac-
tions.
[72] Gautier Izacard, Mathilde Caron, Lucas Hos-
seini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118, 2022.
[73] Benoit Jacob, Skirmantas Kligys, Bo Chen, Men-
glong Zhu, Matthew Tang, Andrew Howard,
Hartwig Adam, and Dmitry Kalenichenko. Quanti-
zation and training of neural networks for efficient
integer-arithmetic-only inference.Proceedings of
the IEEE Conference on Computer Vision and Pat-
tern Recognition, pages 2704–2713, 2018.
[74] Mojan Javaheripi, S ´ebastien Bubeck, Marah
Abdin, Jyoti Aneja, S ´ebastien Bubeck, Caio
C´esar Teodoro Mendes, Allie Del Giorno,
Ronen Eldan, Sivakanth Gopi, Ece Kamar,
et al. Phi-2: The surprising power of small
language models.Microsoft Research Blog,
2023. Available at https://www.microsoft.com/en-
us/research/blog/phi-2-the-surprising-power-of-
small-language-models/.
[75] Albert Q Jiang, Alexandre Sablayrolles, Arthur
Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Bressand, Gi-
anna Lengyel, Guillaume Lample, Lucile Saulnier,
et al. Mistral 7b.arXiv preprint arXiv:2310.06825,
2023.
[76] Albert Q Jiang, Alexandre Sablayrolles, Antoine
Roux, Arthur Mensch, Blanche Savary, Chris Bam-
ford, Devendra Singh Chaplot, Diego de las Casas,
Emma Bou Hanna, Florian Bressand, et al. Mix-
tral of experts.arXiv preprint arXiv:2401.04088,
2024.
[77] Carlos E Jimenez, John Yang, Alexander Wettig,
Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. SWE-bench: Can language models
resolve real-world GitHub issues?arXiv preprint
70

arXiv:2310.06770, 2023. Real GitHub issues from
popular repositories, state-of-art resolves 13.8% of
issues.
[78] Jared Kaplan, Sam McCandlish, Tom Henighan,
Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario
Amodei. Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361, 2020.
[79] Angelos Katharopoulos, Apoorv Vyas, Nikolaos
Pappas, and Franc ¸ois Fleuret. Transformers are
rnns: Fast autoregressive transformers with linear
attention.arXiv preprint arXiv:2006.16236, 2020.
[80] Yaniv Leviathan, Matan Kalman, and Yossi Ma-
tias. Fast inference from transformers via specu-
lative decoding.arXiv preprint arXiv:2211.17192,
2023.
[81] Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, et al. Retrieval-augmented generation
for knowledge-intensive nlp tasks.Advances in
Neural Information Processing Systems, 33:9459–
9474, 2020.
[82] Xiang Lisa Li and Percy Liang. Prefix-tuning: Op-
timizing continuous prompts for generation.arXiv
preprint arXiv:2101.00190, 2021.
[83] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang,
Xingyu Dang, and Song Han. Awq: Activation-
aware weight quantization for llm compression and
acceleration.arXiv preprint arXiv:2306.00978,
2023.
[84] Haotian Liu, Chunyuan Li, Qingyang Wu, and
Yong Jae Lee. Visual instruction tuning.arXiv
preprint arXiv:2304.08485, 2023.
[85] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xu-
anyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding,
Kaiwen Men, Kejuan Yang, et al. AgentBench:
Evaluating LLMs as agents.arXiv preprint
arXiv:2308.03688, 2023. 8 environments spanning
web navigation, database querying, game playing,
household tasks; GPT-4: 52% average.[86] Kevin Meng, David Bau, Alex Andonian, and
Yonatan Belinkov. Locating and editing factual as-
sociations in GPT.Advances in Neural Informa-
tion Processing Systems, 35:17359–17372, 2022.
ROME: Rank-One Model Editing for precise fac-
tual updates in transformer models.
[87] Kevin Meng, Arnab Sen Sharma, Alex Ando-
nian, Yonatan Belinkov, and David Bau. Mass-
editing memory in a transformer.arXiv preprint
arXiv:2210.07229, 2023. MEMIT: Extends ROME
to batch editing multiple facts simultaneously.
[88] Gr ´egoire Mialon, Roberto Dess `ı, Maria Lomeli,
Christoforos Nalmpantis, Ram Pasunuru, Roberta
Raileanu, Baptiste Rozi `ere, Timo Schick, Jane
Dwivedi-Yu, Asli Celikyilmaz, et al. GAIA: A
benchmark for general AI assistants.arXiv preprint
arXiv:2311.12983, 2023. Multi-modal, multi-step
assistant tasks requiring web search, code execu-
tion, file processing; only 15% solvable.
[89] Microsoft Research. Phi-4 reasoning: Small lan-
guage models for complex reasoning.Microsoft
Technical Report, 2025.
[90] MiniMax AI. Minimax-m1: 456b lightning atten-
tion model with partial rope.arXiv preprint, 2025.
[91] Mistral AI. Codestral: A code-specialized large
language model.Mistral AI Blog, 2024. Mistral
7B variant fine-tuned on code from The Stack and
StackOverflow.
[92] Mistral AI. Pixtral 12b.Mistral AI Blog, 2024.
[93] Eric Mitchell, Charles Lin, Antoine Bosse-
lut, Chelsea Finn, and Christopher D Man-
ning. Fast model editing at scale.arXiv
preprint arXiv:2110.11309, 2022. SERAC: Semi-
parametric editing with retrieval and counterfactual
modeling.
[94] Moonshot AI. Kimi 1.5: Long-context reasoning
at scale.arXiv preprint, 2025.
[95] Moonshot AI. Kimi k2: Advanced long-context
language model.arXiv preprint, 2025.
71

[96] Moonshot AI. Kimi linear: 48b parameter model
with hybrid linear attention.arXiv preprint, 2025.
[97] Niklas Muennighoff, Luca Soldaini, Dirk Groen-
eveld, Kyle Lo, Iz Beltagy, et al. Olmoe:
Open mixture-of-experts language models.arXiv
preprint arXiv:2409.02060, 2024.
[98] NVIDIA. Llama-nemotron: Efficient instruction
following through distillation.NVIDIA Technical
Report, 2025.
[99] NVIDIA. Nemotron 3 nano: Ultra-efficient lan-
guage models for edge devices.NVIDIA Technical
Report, 2025.
[100] NVIDIA. Nemotron nano 2: Edge deployment of
large language models.NVIDIA Technical Report,
2025.
[101] Open-Reasoner Team. Open-reasoner-zero: Scal-
ing reasoning through pure reinforcement learning.
arXiv preprint, 2025.
[102] OpenAI. Chatgpt plugins and code interpreter.
OpenAI Blog, 2023. Introduces Code Interpreter
providing sandboxed Python execution environ-
ment for data analysis.
[103] OpenAI. Function calling and other api updates.
OpenAI Blog, 2023. Introduces structured func-
tion calling for GPT-4, enabling precise API inter-
actions.
[104] OpenAI. GPT-4 system card.OpenAI Technical
Report, 2023. Official system card detailing GPT-4
capabilities, limitations, and safety evaluations.
[105] OpenAI. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
[106] OpenAI. Gpt-4o system card.OpenAI Technical
Report, 2024.
[107] OpenAI. Learning to reason with llms.OpenAI
Blog, 2024.
[108] OpenAI Sora Team. Video generation models as
world simulators.OpenAI Technical Report, 2024.[109] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo
Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama,
Alex Ray, et al. Training language models to
follow instructions with human feedback.Ad-
vances in Neural Information Processing Systems,
35:27730–27744, 2022.
[110] Joon Sung Park, Joseph C O’Brien, Carrie J
Cai, Meredith Ringel Morris, Percy Liang, and
Michael S Bernstein. Generative agents: Interac-
tive simulacra of human behavior.arXiv preprint
arXiv:2304.03442, 2023.
[111] Shishir G Patil, Tianjun Zhang, Xin Wang, and
Joseph E Gonzalez. Gorilla: Large language model
connected with massive APIs.arXiv preprint
arXiv:2305.15334, 2023. Fine-tuned for ML API
calls with focus on HuggingFace, PyTorch, Tensor-
Flow libraries.
[112] Jonas Pfeiffer, Aishwarya Kamath, Andreas
R¨uckl´e, Kyunghyun Cho, and Iryna Gurevych.
Adapterfusion: Non-destructive task composition
for transfer learning. InProceedings of the 16th
Conference of the European Chapter of the Asso-
ciation for Computational Linguistics, pages 487–
503, 2020. Composable adapter modules for multi-
task learning.
[113] Chen Qian, Xin Cong, Cheng Yang, Weize
Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu,
and Maosong Sun. ChatDev: Communicative
agents for software development.arXiv preprint
arXiv:2307.07924, 2023. Simulates software com-
pany with 7-role waterfall development, generates
complete software in ¡10 minutes.
[114] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu,
Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xian-
gru Tang, Bill Qian, et al. ToolLLM: Facilitating
large language models to master 16000+ real-world
APIs.arXiv preprint arXiv:2307.16789, 2023.
Large-scale tool learning with 16,000+ real-world
APIs and comprehensive benchmarking.
[115] Qualcomm AI Research. Quantization-aware train-
ing for deep neural networks.https://www.
72

qualcomm.com/developer/ai, 2024. QAT
techniques for neural network quantization.
[116] Alec Radford, Jong Wook Kim, Chris Hallacy,
Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin,
Jack Clark, et al. Learning transferable visual mod-
els from natural language supervision. InInter-
national Conference on Machine Learning, pages
8748–8763. PMLR, 2021.
[117] Alec Radford, Jeffrey Wu, Rewon Child, David
Luan, Dario Amodei, Ilya Sutskever, et al. Lan-
guage models are unsupervised multitask learners.
OpenAI blog, 1(8):9, 2019.
[118] Rafael Rafailov, Archit Sharma, Eric Mitchell, Ste-
fano Ermon, Christopher D Manning, and Chelsea
Finn. Direct preference optimization: Your lan-
guage model is secretly a reward model.arXiv
preprint arXiv:2305.18290, 2023.
[119] Rafael Rafailov, Archit Sharma, Eric Mitchell, Ste-
fano Ermon, Christopher D Manning, and Chelsea
Finn. Direct preference optimization: Your lan-
guage model is secretly a reward model.Ad-
vances in Neural Information Processing Systems,
36, 2023. DPO: optimize policies directly from hu-
man preferences without explicit reward modeling,
simpler than RLHF.
[120] Colin Raffel, Noam Shazeer, Adam Roberts,
Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter J Liu. Exploring
the limits of transfer learning with a unified text-
to-text transformer.Journal of Machine Learning
Research, 21(140):1–67, 2020.
[121] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol,
Casey Chu, and Mark Chen. Hierarchical text-
conditional image generation with clip latents. In
arXiv preprint arXiv:2204.06125, 2022.
[122] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh,
Scott Gray, Chelsea V oss, Alec Radford, Mark
Chen, and Ilya Sutskever. Zero-shot text-to-image
generation. InInternational Conference on Ma-
chine Learning, pages 8821–8831. PMLR, 2021.[123] Sebastian Raschka. Understanding 2025 llm archi-
tectures: A comprehensive guide.Blog post, se-
bastianraschka.com, 2025.
[124] Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy Lillicrap, Jean-Baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan
Firat, Julian Schrittwieser, et al. Gemini 1.5:
Unlocking multimodal understanding across mil-
lions of tokens of context.arXiv preprint
arXiv:2403.05530, 2024.
[125] David Rein, Betty Li Hou, Asa Cooper Stick-
land, Jackson Petty, Richard Yuanzhe Pang, Julien
Dirani, Julian Michael, and Samuel R Bowman.
Gpqa: A graduate-level google-proof q&a bench-
mark.arXiv preprint arXiv:2311.12022, 2023.
[126] Robin Rombach, Andreas Blattmann, Dominik
Lorenz, Patrick Esser, and Bj ¨orn Ommer. High-
resolution image synthesis with latent diffusion
models. InProceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition,
pages 10684–10695, 2022.
[127] Andreas R ¨uckl´e, Gregor Geigle, Max Glockner,
Tilman Beck, Jonas Pfeiffer, Nils Reimers, and
Iryna Gurevych. Adapterdrop: On the efficiency
of adapters in transformers. InProceedings of the
2021 Conference on Empirical Methods in Natu-
ral Language Processing, pages 7930–7946, 2021.
Dynamic adapter selection for improved efficiency.
[128] Chitwan Saharia, William Chan, Saurabh Sax-
ena, Lala Li, Jay Whang, Emily Denton, Seyed
Kamyar Seyed Ghasemipour, Burcu Karagol Ayan,
S Sara Mahdavi, Rapha Gontijo Lopes, et al.
Photorealistic text-to-image diffusion models with
deep language understanding.arXiv preprint
arXiv:2205.11487, 2022.
[129] Rylan Schaeffer, Brando Miranda, and Sanmi
Koyejo. Are emergent abilities of large lan-
guage models a mirage?arXiv preprint
arXiv:2304.15004, 2023.
[130] Timo Schick, Jane Dwivedi-Yu, Roberto
Dess `ı, Roberta Raileanu, Maria Lomeli, Luke
73

Zettlemoyer, Nicola Cancedda, and Thomas
Scialom. Toolformer: Language models can
teach themselves to use tools.arXiv preprint
arXiv:2302.04761, 2023.
[131] John Schulman, Filip Wolski, Prafulla Dhari-
wal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms.arXiv preprint
arXiv:1707.06347, 2017.
[132] Seed-AI. Seed-thinking 1.5: Combining monte
carlo tree search with reinforcement learning for
reasoning.arXiv preprint, 2025.
[133] Jaime Sevilla, Lennart Heim, Anson Ho, Tamay
Besiroglu, Marius Hobbhahn, and Pablo Villalo-
bos. Compute trends across three eras of machine
learning.arXiv preprint arXiv:2202.05924, 2024.
[134] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin
Xu, Junxiao Song, Mingchuan Zhang, YK Li,
Y Wu, and Daya Guo. Deepseekmath: Pushing the
limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, 2024.
[135] Noam Shazeer. Fast transformer decoding: One
write-head is all you need.arXiv preprint
arXiv:1911.02150, 2019.
[136] Noam Shazeer. Glu variants improve transformer.
arXiv preprint arXiv:2002.05202, 2020.
[137] Noam Shazeer, Azalia Mirhoseini, Krzysztof
Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,
and Jeff Dean. Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer.arXiv
preprint arXiv:1701.06538, 2017.
[138] Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learn-
ing. InAdvances in Neural Information Processing
Systems, volume 36, pages 8634–8652, 2023. In-
troduces self-reflection mechanism enabling agents
to learn from failures through iterative improve-
ment.
[139] Skywork AI. Skywork-o1: Open reasoning with
verifiable rewards.arXiv preprint, 2025.[140] Reid G Smith. The contract net protocol: High-
level communication and control in a distributed
problem solver.IEEE Transactions on computers,
29(12):1104–1113, 1980. Contract Net Protocol:
agent task assignment via bidding and negotiation.
[141] Daria Soboleva, Rami Al-Rfou, and Aarohi Srivas-
tava. GPT-4 technical report commentary: Nav-
igating the unknowns of advanced AI.arXiv
preprint arXiv:2303.12712, 2023. Analysis and
commentary on GPT-4 capabilities and limitations.
[142] Aarohi Srivastava, Abhinav Rastogi, Abhishek
Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam
Fisch, Adam R Brown, Adam Santoro, Aditya
Gupta, Adri `a Garriga-Alonso, et al. Beyond the
imitation game: Quantifying and extrapolating the
capabilities of language models.arXiv preprint
arXiv:2206.04615, 2022.
[143] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel
Ziegler, Ryan Lowe, Chelsea V oss, Alec Radford,
Dario Amodei, and Paul F Christiano. Learning
to summarize from human feedback.Advances in
Neural Information Processing Systems, 33:3008–
3021, 2020.
[144] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and
Yunfeng Liu. Roformer: Enhanced transformer
with rotary position embedding.arXiv preprint
arXiv:2104.09864, 2021.
[145] Yutao Sun, Li Dong, Shaohan Huang, Shuming
Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and
Furu Wei. Retentive network: A successor to trans-
former for large language models.arXiv preprint
arXiv:2307.08621, 2023.
[146] Nathan J Szymanski, Bernardus Rendy, Yuxing
Fei, Rishi E Kumar, Tanjin He, David Milsted,
Matthew J McDermott, Max Gallant, Ekin Dogus
Cubuk, Amil Merchant, et al. An autonomous lab-
oratory for the accelerated synthesis of novel ma-
terials.Nature, 624(7990):86–91, 2023. A-Lab:
autonomous chemistry laboratory for materials dis-
covery.
[147] Julian Taylor, Wenxuan Hu, Aswathy Guo, et al.
Fine-tuning large language models for domain
74

adaptation: Exploration of training strategies, scal-
ing, model merging and synergistic capabilities.
npj Computational Materials, 11:6, 2025.
[148] Hugo Touvron, Thibaut Lavril, Gautier Izacard,
Xavier Martinet, Marie-Anne Lachaux, Timoth ´ee
Lacroix, Baptiste Rozi `ere, Naman Goyal, Eric
Hambro, Faisal Azhar, et al. Llama: Open and ef-
ficient foundation language models.arXiv preprint
arXiv:2302.13971, 2023.
[149] Hugo Touvron, Louis Martin, Kevin Stone, Peter
Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. Llama 2: Open founda-
tion and fine-tuned chat models.arXiv preprint
arXiv:2307.09288, 2023.
[150] Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is
all you need. InAdvances in Neural Information
Processing Systems, pages 5998–6008, 2017.
[151] Pablo Villalobos, Jaime Sevilla, Lennart Heim,
Tamay Besiroglu, Marius Hobbhahn, and Anson
Ho. Will we run out of data to train large language
models?Epoch AI Research, 2022.
[152] Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533, 2024. E5 embeddings: high-
quality dense retrieval model trained on diverse
corpora.
[153] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc
Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves
chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2023.
[154] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel,
Barret Zoph, Sebastian Borgeaud, Dani Yogatama,
Maarten Bosma, Denny Zhou, Donald Metzler,
et al. Emergent abilities of large language models.
arXiv preprint arXiv:2206.07682, 2022.[155] Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi,
Quoc Le, and Denny Zhou. Chain-of-thought
prompting elicits reasoning in large language mod-
els.arXiv preprint arXiv:2201.11903, 2022.
[156] Michael Wooldridge.An introduction to multiagent
systems. John Wiley & Sons, 2nd edition, 2009.
Foundational textbook defining agent properties:
autonomy, reactivity, proactivity, social ability.
[157] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yi-
ran Wu, Shaokun Zhang, Erkang Zhu, Beibin
Li, Li Jiang, Xiaoyun Zhang, and Chi Wang.
AutoGen: Enabling next-gen LLM applications
via multi-agent conversation.arXiv preprint
arXiv:2308.08155, 2023. Framework for conversa-
tional multi-agent systems with customizable roles
and human-in-loop.
[158] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu,
Julien Demouth, and Song Han. Smoothquant: Ac-
curate and efficient post-training quantization for
large language models.International Conference
on Machine Learning (ICML), 2023. Efficient
quantization method balancing accuracy and per-
formance.
[159] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu,
Julien Demouth, and Song Han. SmoothQuant:
Accurate and efficient post-training quantization
for large language models.International Confer-
ence on Machine Learning, pages 38087–38099,
2023. Migrates quantization difficulty from activa-
tions to weights through mathematically equivalent
transformations.
[160] Xiaomi AI Lab. Mimo: Efficient language model
training.arXiv preprint, 2025.
[161] Xiaomi AI Lab. Mimo-v2-flash: High-speed in-
ference with speculative decoding.arXiv preprint,
2025.
[162] Xiaomi AI Lab. Mimo vl: Multimodal understand-
ing and generation.arXiv preprint, 2025.
[163] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng,
Shuxin Zheng, Chen Xing, Huishuai Zhang,
75

Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On
layer normalization in the transformer architecture.
arXiv preprint arXiv:2002.04745, 2020.
[164] Zheyu Yang, Siyuan Liu, Xiang Wang, et al. Will
LLMs scaling hit the wall? breaking barriers
via distributed resources on massive edge devices.
arXiv preprint arXiv:2503.08223, 2025. Available
at https://arxiv.org/html/2503.08223v1.
[165] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Thomas L Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem
solving with large language models.arXiv preprint
arXiv:2305.10601, 2023.
[166] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. Re-
act: Synergizing reasoning and acting in language
models.arXiv preprint arXiv:2210.03629, 2023.
[167] Zifei Yao, Xiaoxia Wu, Chun-Fu Li, et al. Scal-
ing laws for post-training quantized large language
models.arXiv preprint arXiv:2410.12119, 2024.
[168] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Lu-
ong, Gunjan Baid, Zirui Wang, Vijay Vasudevan,
Alexander Ku, Yinfei Yang, Burcu Karagol Ayan,
et al. Scaling autoregressive models for content-
rich text-to-image generation. InarXiv preprint
arXiv:2206.10789, 2022.
[169] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng,
Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, et al.
Judging llm-as-a-judge with mt-bench and chatbot
arena.arXiv preprint arXiv:2306.05685, 2023.
[170] Zhipu AI. Glm-4.5: All tools integration for agen-
tic ai.arXiv preprint, 2025.
[171] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui
Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng,
Yonatan Bisk, Daniel Fried, Uri Alon, et al.
WebArena: A realistic web environment for
building autonomous agents.arXiv preprint
arXiv:2307.13854, 2023. Benchmark for web nav-
igation tasks (e-commerce, forums, CMS), GPT-4
achieves 14.4% success.[172] Ligeng Zhu, Zhijian Liu, and Song Han. Deep
leakage from gradients.Advances in Neural In-
formation Processing Systems, 32, 2019. Demon-
strates gradient inversion attacks that can recon-
struct training samples from shared gradients with
high accuracy.
76