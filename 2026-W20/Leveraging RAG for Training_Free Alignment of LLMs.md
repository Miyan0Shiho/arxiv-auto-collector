# Leveraging RAG for Training-Free Alignment of LLMs

**Authors**: John T. Halloran

**Published**: 2026-05-11 20:29:39

**PDF URL**: [https://arxiv.org/pdf/2605.11217v1](https://arxiv.org/pdf/2605.11217v1)

## Abstract
Large language model (LLM) alignment algorithms typically consist of post-training over preference pairs. While such algorithms are widely used to enable safety guardrails and align LLMs with general human preferences, we show that state-of-the-art alignment algorithms require significant computational resources while being far less capable of enabling refusal guardrails for recent agentic attacks. Thus, to improve refusal guardrails against such attacks without drastically increasing computational overhead, we introduce Retrieval Augmented Generation for Pref erence alignment (RAG-Pref), a simple RAG-based alignment algorithm which conditions on preferred and dispreferred samples to leverage contrastive information during inference. RAG-Pref is online (training-free), compatible with off-the-shelf packages, and, when combined with offline (training-based) alignment algorithms, enables more than an average 3.7 factor improvement in agentic attack refusals across five widely used LLMs, compared to 2.9 for other online alignment algorithms and 1.5 for offline alignment alone. We conclude by showing that, in stark contrast to other online alignment methods, RAG-Pref similarly increases performance on general human-preference alignment tasks and does not drastically increase overall computational requirements.

## Full Text


<!-- PDF content starts -->

Leveraging RAG for Training-Free Alignment of LLMs
John Halloran∗
Leidos
halloranjt@leidos.com
Abstract
Large language model (LLM) alignment algorithms typically consist of post-training over preference pairs. While
such algorithms are widely used to enable safety guardrails and align LLMs with general human preferences, we show
that state-of-the-art alignment algorithms require significant computational resources while being far less capable of
enabling refusal guardrails for recent agentic attacks. Thus, to improve refusal guardrails against such attacks without
drastically increasing computational overhead, we introduceRetrievalAugmentedGeneration forPreference alignment
(RAG-Pref), a simple RAG-based alignment algorithm which conditions on preferred and dispreferred samples to leverage
contrastive informationduring inference. RAG-Pref is online (training-free), compatible with off-the-shelf packages, and,
when combined with offline (training-based) alignment algorithms, enables more than an average 3.7 factor improvement
in agentic attack refusals across five widely used LLMs, compared to 2.9 for other online alignment algorithms and 1.5 for
offline alignment alone. We conclude by showing that, in stark contrast to other online alignment methods, RAG-Pref
similarly increases performance on general human-preference alignment tasks and does not drastically increase overall
computational requirements.
1 Introduction
Alignment has become a critical step towards ensuring large language model (LLM) responses align with general hu-
man preferences . Currently, alignment algorithms are dominated by reinforcement learning-based schemes–such as
RLHF (Ouyang et al., 2022) and the more efficient direct preference optimization (DPO) (Rafailov et al., 2023)–wherein
models are post-trained over pairs of preferred and dispreferred responses for each input query. The responses from result-
ing models are thus aligned with desirable behaviors present in preferred training data, while the undesirable behaviors
present in dispreferred training data are avoided. Suchalignment-tuning–i.e., alignment via RLHF, DPO, or derivative
preference optimization post-training algorithms–has proven pivotal towards producing LLM assistants whose responses
are accurate as well ashelpful(Dubois et al., 2024; Ouyang et al., 2022; Zheng et al., 2023; Zhong et al., 2025).
With the success of alignment-tuning and growing evidence that LLMs are highly susceptible to adversarial attacks (Chao
et al., 2025; Liu et al., 2025, 2023b; Mehrotra et al., 2024; Shen et al., 2024; Zhang et al., 2023), significant works have sought
to produce LLM assistants whose responses are not only helpful, but also aligned withharmlessbehaviors (Bai et al., 2022;
Ji et al., 2023; Tian et al., 2024). Such works have shown that safety alignment-tuning (SAT) may be achieved by directly
including harmless and harmful preference pairs during alignment-tuning (Ji et al., 2023; Team et al., 2023; Tian et al., 2024).
SAT thus enablesrefusal guardrails, whereby models learn to refuse malicious instructions containing harmful language,
while complying with benign requests. Furthermore, recent works have shown that such refusal guardrails may be further
strengthened when additional safety training labels are available via specialized SAT algorithms (Dai et al., 2024; Kim et al.,
2026).
While SAT has become a ubiquitous and critical step when deploying frontier models (Anthropic, 2025; DeepSeek-AI, 2025;
Grattafiori et al., 2024; Hurst et al., 2024; Yang et al., 2025), three major drawbacks have emerged: 1) training requires
significant computational overhead, (2) SAT isoffline, i.e., retraining is required if newly catalogued adversarial attacks
are to be guarded against, and–most importantly–(3) SAT frontier models have proven to be highly susceptible to recent
agentic attacks which lack common refusal triggers (i.e., harmful language). The latter has emerged with the recent advent
of themodel context protocol(MCP), a widely adapted universal agentic protocol (Anthropic, 2025b). In particular, despite
∗Alternative contact: halloj3@uw.edu.
1arXiv:2605.11217v1  [cs.LG]  11 May 2026

extensive SAT (Anthropic, 2025; Grattafiori et al., 2024), MCP-enabled LLMs have proven to be highly susceptible to
adversarial attacks which induce malicious tool use yet lack standard refusal trigger phrases (Radosevich & Halloran, 2025).
Such agentic attacks are referred to herein asfalsely benign attacks(FBAs).
Most concerningly, we show that the effectiveness of FBAs are not simply due to the recency of the MCP and the lack
of such attack data during SAT. Collecting a high-quality dataset of FBAs and truly benign (TB) samples, we safety-tune
five popular LLMs using state-of-the-art (SOTA) SAT algorithms DPO and SafeDPO (Kim et al., 2026). Across all models,
DPO and SafeDPO display limited ability to enable FBA refusal guardrails, improving baseline refusal rates by only an
average factor of 1.4 and 1.6, respectively. Alarmingly, no safety-tuned model achieves an FBA refusal rate greater than 48%.
Thus, to further improve refusal guardrails in the face of agentic attacks, we introduceRetrievalAugmentedGeneration
forPreference alignment (RAG-Pref), a new RAG-based alignment algorithm which conditions on both preferred and
dispreferred examples during inference.
RAG-Pref is online (training-free), easily implementable using off-the-shelf packages/components, and significantly
improves LLM refusal guardrails compared to both SOTA offline and online alignment algorithms. In particular, RAG-Pref
increases baseline refusal rates by an average 3-fold improvement–2.1 and 1.9 times greater than DPO and SafeDPO,
respectively, and 1.8 times greater than recent online alignment strategies (Zhu et al., 2025). Moreover, we show that RAG-
Pref further improves performance when augmenting generation with samples from the LLM’s closed-book (parametric)
knowledge; when combined with DPO and SafeDPO aligned models, RAG-Pref further increases baseline refusal rates by
an average 3.5- and 3.9-fold improvements, respectively.
In addition to significantly improving agentic guardrails, we demonstrate that RAG-Pref similarly improves performance
for general human-preference alignment tasks; across five SOTA alignment-tuned models, RAG-Pref increases performance
over alternative online alignment algorithms by an average 26.2% and 6.4% on standard benchmarks AlpacaEval 2 and
MT-Bench, respectively. Furthermore, we demonstrate the computational advantages of RAG-Pref over offline and other
online alignment algorithms; RAG-Pref requires three-orders of magnitude less preprocessing time and two-orders of
magnitude less GPU memory compared to offline safety-tuning a 14B parameter model, while only incurring an inference
slowdown of 20% compared to 372% for SOTA decoding-based online alignment (Zhu et al., 2025). Finally, we theoretically
show that RAG-Pref is guaranteed to further reduce standard RAG’s expected uncertainty by a nonnegative amount, and
verify this empirically.
2 Background and Related Work
2.1 LLM Alignment-Tuning
LLM alignment most commonly consists of fine-tuning given queries with accompanying preferred and dispreferred
response pairs (which reflect human preferences). Initial alignment approaches, such as RLHF (Ouyang et al., 2022) and
RLAIF (Lee et al., 2024), first trained reward models on human preference data, then utilized reinforcement learning
(RL) fine-tuning to learn a policy which maximized the reward signal. To address training instability in RL fine-tuning,
subsequent work reparameterized the RL objective, allowing direct learning of the optimal policy with a simple closed
form objective (Rafailov et al., 2023). The resulting algorithm (i.e., DPO), was shown to provide significantly better training
stability than RL-based alignment.
For general human-preference tasks, DPO and its many follow-up variants (Chen et al., 2024; Chowdhury et al., 2024;
D’Oosterlinck et al., 2025; Ji et al., 2024; Jung et al., 2024; Liu et al., 2024; Melnyk et al., 2024; Wu et al., 2024), have
been extensively studied and demonstrated SOTA performance. Furthermore, the recent reinforced token optimization
(RTO) (Zhong et al., 2024) has shown that by optimizing over token-wise reward signals, performance on general human-
preference benchmarks (e.g., AlpacaEval 2) may be significantly improved.
For online training-free alignment, (Zhu et al., 2025) introduced On-the-fly Preference Alignment via Principle-Guided
Decoding (OPAD). OPAD calculates a similar reward function as used in DPO to adjust the per-token conditional distribution
during decoding and was shown to improve performance on general preference alignment tasks relative to previous online
methods (Gao et al., 2024). OPAD’s reward function and decoding procedure are further detailed in Section 3.
Safety Alignment-Tuning.Additional works have sought to further focus alignment on safety and decrease the risk
of unsafe behaviors–e.g., toxicity (Hartvigsen et al., 2022), hate speech (Mazeika et al., 2024), and compliance with
malicious/violent requests (Li et al., 2024). (Dai et al., 2024) utilized additional labels (safe/unsafe) for each preferred and
2

dispreferred training pair used in RLHF to derive a three-round fine-tuning algorithm, called Safe RLHF, which optimized
both for preference and safety alignment. Subsequently, (Kim et al., 2026) showed that, in the presence of safe/unsafe
labels for preferred and dispreferred pairs, the DPO objective could be adjusted to simultaneously optimize for safety with
an additional loss offset. The resulting algorithm, called SafeDPO, was shown to offer improved training stability and
safety alignment performance compared to Safe RLHF.
2.2 Agentic Adversarial Attacks
Extensive works have studied the susceptibility of LLMs to attacks which circumvent refusal guardrails for malicious
purposes. Jailbreaks (Chao et al., 2025; Zou et al., 2023) craft input prompts which evade guardrails to elicit unsafe responses.
Prompt injection attacks (PIAs) (Greshake et al., 2023; Liu et al., 2023a; Perez & Ribeiro, 2022) consist of injecting malicious
instructions into user prompts. Encoded prompt injections encode malicious commands in an alternative format (e.g., octal)
for PIAs. However, frontier LLM safety training has grown to include standard LLM attacks (Chao et al., 2024; Mazeika
et al., 2024), thus expanding refusal guardrails to include existing jailbreaks and PIAs (Grattafiori et al., 2024; Sharma et al.,
2025).
Recently, the MCP has seen massive, widespread adaption (Anthropic, 2025a). By standardizing API calls between
LLMs, tools, and data sources, the MCP enables seamless integration between generative AI agents and widely used
applications (Anthropic, 2025c; Google, 2025; Microsoft, 2025; Stripe, 2025). However, distinct from previous LLM jailbreaks
and PIAs, recent work has shown that the MCP introduces new attack possibilities in agentic systems (Radosevich &
Halloran, 2025).
Radosevich & Halloran (2025) demonstrated that MCP-enabled agents are susceptible to PIAs which explicitly lack refusal
guardrail triggers. I.e., while refusal guardrails are triggered by malicious PIAs which explicitly state harmful phrases or
suspicious text, attacks lacking these exact triggers are successfully completed. As previously mentioned, we term such
PIAs which lack standard harmful or suspicious cuesfalsely benign attacks(FBAs). The success of FBAs is attributed to
the shift in attack goals from LLMs–which focus on unsafe text generation, the attacks of which contain related harmful
phrases or suspicious text–to MCP-enabled LLMs–which focus on the malicious execution of tools, the attacks of which
need not contain harmful or suspicious text found in standard SAT data.
Agentic Safety work.Existing MCP defenses have focused on either inspecting MCP-servers for potential vulnerabili-
ties (Halloran, 2025; Invariant, 2025b) or monitoring user queries for suspicious traffic (Invariant, 2025a). However, to the
best of the authors’ knowledge, no previous works have explored the effect of MCP-specific attacks on SAT algorithms and
the performance of resulting refusal guardrails. Furthermore, in contrast to existing work seeking to optimize specific
components in RAG systems–described as retriever-generator alignment (Jin et al., 2025; Sun et al., 2025; Wu et al., 2025)–or
using RAG to generate preference alignment data (Song et al., 2025), the authors are unaware of previous work exploring
online RAG alignment algorithms.
3 Offline and Online Preference Alignment
Let𝑥be an input prompt. For an autoregressive LLM 𝜋𝜃, consider the probability of generating a response 𝑦consisting of
𝑇tokens:
𝜋𝜃(𝑦|𝑥)=𝑇Ö
𝑡=1P𝜋𝜃(𝑦𝑡|𝑥,𝑦 1:𝑡−1),(1)
whereP 𝜋𝜃(𝑦𝑡|𝑥,𝑦 1:𝑡−1)∈R𝐿𝑡×𝑉,𝐿𝑡is the sequence length at time𝑡, and𝑉is the vocabulary size.
LetD={(𝑥1,𝑦1
𝑤,𝑦1
𝑙),...,(𝑥𝑛,𝑦𝑛
𝑤,𝑦𝑛
𝑙)}be a set of preference data where, for input 𝑥𝑖,𝑦𝑖
𝑤is the preferred response, and 𝑦𝑖
𝑙
is the dispreferred response, denoted as 𝑦𝑖
𝑤≻𝑦𝑖
𝑙|𝑥. Let𝜋𝜃be a model to be optimized and 𝜋refa reference model (which is
typically a supervised fine-tuned version of the model trained prior to preference alignment). The goal of offline alignment
with DPO is thus to train a new model by solving
𝜋∗
𝜙=argmax
𝜋𝜙L(𝑟 𝜙),
L(𝑟 𝜙)=−E(𝑥,𝑦 𝑤,𝑦𝑙)∼D
log𝜎 𝛽𝑟𝜙(𝑥,𝑦 𝑤)−𝛽𝑟 𝜙(𝑥,𝑦 𝑙)
,
3

where𝑟𝜙(𝑥,𝑦 𝑤)=log𝜋𝜙(𝑦𝑤|𝑥)
𝜋𝜃(𝑦𝑤|𝑥)is the reward function. DPO thus learns parameters which align with preferred responses,
without drastically diverging from the reference model.
To avoid training, On-the-fly Preference Alignment via Principle-Guided Decoding (OPAD) (Zhu et al., 2025) includes
additional helpfulness instructions 𝑐and adjusts the per-timestep distribution in Equation 1. Rewriting the reward function
at timestep𝑡as𝑟 𝜃(𝑥,𝑦 1:𝑡,𝑐)=log𝜋𝜃(𝑦1:𝑡|𝑥,𝑐)
𝜋𝜃(𝑦1:𝑡|𝑥), the𝑡th token distribution is adjusted during generation as
P𝜋𝜃(𝑦𝑡|𝑥,𝑐,𝑦 1:𝑡)∝𝜋 𝜃(𝑦𝑡|𝑥,𝑐,𝑦 1:𝑡)exp1
𝛽𝑟𝜃(𝑥,𝑦 1:𝑡,𝑐)
.
We note that the above decoding procedure requires both invasive changes to generation (raising potential compatibility
issues with deployed models) and a significant increase in computational resources–i.e., per time-step, calculation of the
reward function and the subsequent distribution update requires maintaining three separate distributions.
3.1 Training-free Alignment with RAG-Pref
We now detail how online preference alignment may be performed without requiring any invasive adjustments to model
generation via RAG-Pref. Let 𝑒(·)∈R𝑚be a text embedding function (trained to embed semantically similar text near
one another in the embedding space), and 𝑑(·,·) ∈R be a vector-distance metric. For our preference dataset D, let
D𝑤={𝑦 𝑤:(𝑥,𝑦 𝑤,𝑦𝑙)∈D} andD𝑙={𝑦 𝑙:(𝑥,𝑦 𝑤,𝑦𝑙)∈D} be the sets of preferred and dispreferred text responses,
respectively, and let D𝑤
𝑒={𝑒(𝑦) :𝑦∈D𝑤}andD𝑙={𝑒(𝑦) :𝑦∈D𝑙}be the vector databases of preferred and dispreferred
embeddings, respectively.
Algorithm 1RAG-Pref for online alignment.
Input:Query𝑥,D𝑤
𝑒,D𝑙
𝑒, and number of retrieval elements𝑘.
1:Embed𝑥,𝑥′=𝑒(𝑥).
2:For all𝑧∈D𝑤
𝑒, rank each element by𝑑(𝑥′,𝑧), sort, and return the top𝑘sequencesZ𝑤⊆D𝑤
3:For all𝑧∈D𝑙
𝑒, rank each element by𝑑(𝑥′,𝑧), sort, and return the top𝑘sequencesZ𝑙⊆D𝑙
4:Create an instruction that model responses follow retrieved preference instances and avoid dispreferred instances,
denoted asZ𝑤≻Z𝑙.
5:return𝜋 𝜃(𝑦|𝑥,Z𝑤≻Z𝑙).
The RAG-Pref algorithm is detailed in Algorithm 1. Compared to alternative alignment algorithms, we note that:
•DPO, and related offline algorithms, enforce preference alignment over response pairs (𝑦𝑤,𝑦𝑙)to a given query. In
contrast, RAG-Pref enables preference alignment over sets of responses,Z𝑤andZ𝑙.
•No invasive or complicated changes to generation are required, thus allowing compatibility with widely supported
packages and widely used models.
3.2 RAG-Pref Encodes Contrastive Information
We now prove theoretical results for both RAG and RAG-Pref. For the sets of all queries X, all responsesY, and all retrieval
documentsD, let𝑋∈X and𝑌∈Y be a random query and response, respectively, and let 𝑍𝑤,𝑍𝑙∈D be random retrieved
preference and dispreference documents, respectively. Note that, for simplicity, we overload the term document; when
𝑘≥1,𝑍𝑤and𝑍𝑙represent the respective 𝑘-shot demonstrations concatenated together in Algorithm 1, which, without
loss of generality, may each be represented as a document inDwhich has been retrieved.
Firstly, for the distribution returned in Algorithm 1– 𝜋𝜃(𝑌|𝑋,𝑍𝑤≻𝑍𝑙)=𝜋 𝜃(𝑌|𝑋,𝑍𝑤,𝑍𝑙)–we note that RAG-Pref performs
contrastive conditioning, i.e., it conditions generation not only on positive examples (what to do) but also on negative
examples (what to avoid). Furthermore, standard RAG lacks this contrastive conditioning, and thus does not condition
generation on behaviors to avoid.
4

Contrastive conditioning allows us to exactly quantify the reduction in uncertainty provided by RAG-Pref compared to
standard RAG. Let Δ𝐻RAG=𝐼(𝑌 ;𝑍𝑤|𝑋)andΔ𝐻RAG-Pref =𝐼(𝑌 ;𝑍𝑤,𝑍𝑙|𝑋), where𝐼(·)is the mutual information. We define
thecontrastive informationto be Δ𝐻RAG-Pref−Δ𝐻 RAG, which is the amount of additional expected information provided
by Algorithm 1 over standard RAG. In the following, we prove the that the contrastive information is guaranteed to be
nonnegative:
Theorem 3.1. Δ𝐻 RAG-Pref≥Δ𝐻 RAG. Furthermore, when dispreferred examples provide non-redundant information compared
to preferred examples,Δ𝐻 RAG-Pref >Δ𝐻 RAG.
The proof of Theorem 3.1 is available in Appendix A. We note that for safety guardrail alignment, attack patterns are often
semantically distinct from refusal responses, thus providing substantial contrastive information. Furthermore, Theorem 3.1
explains recent empirical findings, wherein standard RAG was shown to degrade LLM refusal guardrails (An et al., 2025);
standard RAG candecreasesafety by retrieving attack examples without contrastive refusal examples, causing the model to
misinterpret these as behaviors to follow rather than refuse.
Finally, we show the following related result.
Theorem 3.2.The maximum reduction in uncertainty between standard inference and RAG/RAG-Pref is lower-bounded by
the contrastive information.
The proof of Theorem 3.2 is available in Appendix B. This result thus shows the total reduction in conditional entropy from
standard inference and RAG/RAG-Pref can be no lower than the contrastive information encoded by RAG-Pref.
4 FBA safety alignment data
FBAs were obtained by mapping an extensive catalog of known exploits to the sequence of MCP tools capable of achieving
the exploit. Herein, we consider 10 tools which equip agents with Linux-like file/directory manipulation abilities (listed in
Table 5). Attacks were obtained from the Common Vulnerabilities and Exposures (CVEs) (CVE Project, 2023) catalog, an
up-to-date corpus of cyber attacks and exploits. The CVEs corpus was filtered for the agentic attacks used in (Radosevich
& Halloran, 2025)–malicious code execution, remote access control, and credential theft–resulting in ∼34k samples. Using
gpt-4o, each CVE attack was then: a) mapped to a sequence of Linux commands, (b) marked as feasible or not given the
set of MCP tools, (c) feasible attacks were mapped to sequences of MCP tool calls, and (d) friendly malicious requests
(i.e., FBAs) were generated given the original CVE goal. This resulted in 1,150 FBA dispreferred responses, where friendly
malicious requests were queries and the respective MCP tool calls were responses. For each FBA, preferred responses were
set to a direct refusal.
TB samples were collected usingClaudeto create useful examples per MCP tool while assuming specific roles (e.g., system
admin, AI researcher, etc.), thus generating TB queries and preferred responses. TB dispreferred responses were created by
setting the tools used during completion to their opposite (e.g., read_file substituted with write_file ). All TB samples
were verified by hand. The final dataset consists of 1,035 training FBAs, 1,035 training TB samples, and 115 FBA testing
samples. Further pipeline details are available in Appendix C.
We note that this dataset is designed to enable refusal guardrails given paired FBA and TB data, wherein FBA preferred
samples are direct refusals, FBA dispreferred samples are attacks based on MCP tool calls, TB preferred samples are benign
MCP tool calls, and TB dispreferred samples are benign (but incorrect) MCP tool calls.The abilityfor each FBA’s sequence
of malicious MCP tool calls–as well as each SAT model–to exactly achieve the original CVE attack is outside the scope
of this work, as is the ability to accurately assess and report FBA attack success rates. Rather,we focus on the abilityof
SOTA SAT algorithmsto enable refusal guardrails against FBAs, and thus report FBA refusal scores.
5 Experiments
FBA refusal guardrails.We consider five widely used open-source LLMs, varying in parameter count from 1B to 14B.
RAG-Pref is compared to offline (DPO and SafeDPO) and online (OPAD) alignment methods. DPO and SafeDPO refusal
alignment were performed using FBA and TB preference pairs (described in Section 4). RAG-Pref preferred and dispreferred
vector databases were formed from the same preference pairs used for offline alignment. In addition to base models,
5

Table 1:FBA Refusal Rates for Offline and Online Aligned Models: Refusal rates calculated over the test FBAs. Bold
= highest per model.Gemma-2-2B-ITincompatible with OPAD. “B” is base model, “D” is DPO SAT, and “S” is SafeDPO SAT
performance.
Offline Only + OPAD + RAG-Pref
Model B D S B D S B D S
Gemma-2-2B-IT0.32 0.45 0.47 – – – 0.63 0.740.75
Llama-3.1-8B-Instruct0.35 0.43 0.45 0.43 0.47 0.37 0.950.97 0.97
Llama-3.2-1B-Instruct0.15 0.31 0.40 0.59 0.61 0.66 0.28 0.580.88
DeepSeek-R1-Distill-Llama-8B0.14 0.15 0.13 0.44 0.47 0.45 0.59 0.590.59
DeepSeek-R1-Distill-Qwen-14B0.16 0.18 0.19 0.41 0.44 0.46 0.640.680.64
RAG-Pref and OPAD were combined with offline aligned models. For each method, ten generations were assessed per test
instance and average refusal rate reported. Further experimental details are discussed in Appendix 4.
Owing to the use of off-the-shelf components, RAG-Pref was compatible with all evaluated models. In stark contrast,
OPAD’s invasive decoding scheme was incompatible withGemma-2-2B-IT(including the original unbatched version, a
compute-optimized version developed for timing comparisons, and extensively debugged variants).
Refusal rates were calculated using the test FBA data described in Section 4. Refusal rates across all methods are listed in
Table 1. Despite the majority of evaluated base models undergoing excessive post-training safety alignment (DeepSeek-AI,
2025; Gemma et al., 2024; Grattafiori et al., 2024), no base model achieves an FBA refusal rate over 35%. Furthermore, while
refusal rates improve given offline alignment, neither DPO nor SafeDPO enable refusal rates beyond 48%. Thus,SOTA
offline alignment methods provide limited refusal guardrails against FBAs.
RAG-Pref successfully improves guardrails over all base and SAT models, while OPAD fails to improveLlama-3.1-8B-
Instruct SafeDPOguardrails (yet succeeds on the other models). OPAD outperforms RAG-Pref for the base and DPO-
alignedLlama-3.2-1B-Instructmodels. However, RAG-Pref greatly outperforms OPAD all other model configurations,
providing an average 50% more refusal performance across the twelve models. Furthermore, for the models OPAD was
unable to align, RAG-Pref improved refusal guardrails by an average 74%. Thus,RAG-Pref drastically outperforms
other online methods for the refusal of FBAs, while significantly improving the refusal guardrails of SOTA
offline alignment methods.
Due to space constraints, RAG-Pref is compared to vanilla RAG in Figure 3. Across all base and offline aligned models,
RAG-Pref achieves significantly higher refusal rates than standard RAG, resulting in an average 2.8-fold safety improvement
over all models. Furthermore, while RAG-Pref uniformly improves refusal rates for all models, standard RAG actually
decreases refusal performance forLlama-3.1-8B-InstructandDeepSeek-R1-Distill-Llama-8Bmodels, reconfirming the
results of other recent work (An et al., 2025).
Example safety generations are displayed in Appendix I. Furthermore, we ablate several hyperparameters used for offline
SAT methods:
•SAT loss function:In Figure 1, we align Llama-3.2-1B using 10 different DPO loss functions, exploring the effect
of the DPO loss on refusal alignment. The default “sigmoid” loss, used for all other experiments herein, achieves the
highest refusal rate (31.4%).
•Number of SAT epochs:In Figure 2, we increase the number of DPO training epochs to 90 (4 fold increase) for
DeepSeek-R1-Distill-Qwen-14B. Training quickly converges within the original training recipe (15 epochs, i.e.,
15,000 steps) in Figure 2a.
Computational ComparisonWe compare the total training/preparation time, inference time, and inference memory
usage of online and offline methods for refusal alignment and evaluation ofDeepSeek-R1-Distill-Qwen-14B. Training
data used was the 4,410 TB/FBA preference pairs and evaluation data (for inference) was the 115 FBA test samples. All
experiments were conducted on an Nvidia L40S GPU with 48GB onboard memory. The batch size for training (DPO),
vector database preparation (RAG-Pref), and inference (all methods) was maximized for each method given GPU memory.
6

The original OPAD codebase was written for single-sample inference, and was subsequently optimized using batched
inference and extensive vectorization for a fair computational comparison. Results are reported in Table 2.
Table 2: Inference/training runtimes and inference per-batch memory usage for offline and online refusal alignment of
DeepSeek-R1-Distill-Qwen-14B.
Method Train/Prep↓Inf.↓GPU Mem/Batch↓
(hrs) (hrs) (GB)
DPO13.31.4 1.3
RAG-Pref1.7×10−31.8 1.7
OPAD 05.4 7.2
Table 3:Human-Preference Alignment Performance:AlpacaEval 2 and MT-Bench results. For each aligned model and
task, top performing online alignment is highlighted in bold.
Offline Alignment
Metric Online Alignment SFT DPO PPO SimPO RTO
AlpacaEval 2 (WR)RAG-Pref10.87 19.50 19.07 18.14 34.29
RAG 10.37 16.83 17.76 14.66 32.67
OPAD 0.99 5.59 7.95 7.95 9.62
– 9.44 12.86 14.66 10.56 31.30
AlpacaEval 2 (LC)RAG-Pref14.48 24.82 28.59 23.18 37.45
RAG 13.26 21.81 27.78 19.66 36.56
OPAD 2.26 7.61 10.16 8.41 10.69
– 14.17 14.46 18.30 16.85 36.17
MT-Bench (SAG)RAG-Pref6.01 6.19 6.49 5.84 6.83
RAG 5.97 6.12 6.45 5.74 6.75
OPAD 3.58 4.05 5.14 3.16 4.58
– 5.74 6.00 6.22 5.84 6.54
RAG-Pref preprocessing time is 7,824 times faster than DPO offline alignment. Furthermore, RAG-Pref does not add
significant inference overhead compared to other online alignment methods, i.e., RAG-Pref is 3 times faster than OPAD,
while also requiring 4.2 times less memory per instance. Thus, RAG-Pref preprocessing is substantially faster than offline
alignment, while RAG-Pref inference is significantly more efficient than other online alignment methods.
5.1 General Human-Preference Alignment Tasks
For standard general human-preference alignment benchmarks AlpacaEval 2 (Dubois et al., 2024) and MT-Bench (Zheng
et al., 2023), we present the performance of online (OPAD, RAG, and RAG-Pref) and SOTA offline alignment algorithms.
These benchmarks test a model’s conversational and multi-turn ability to generate responses which align with human
preferences. Supervised fine-tuning (SFT) was first performed on a base Llama3-8B model, followed by separate offline
preference alignment algorithms DPO, PPO (for RLHF), SimPO, and RTO.
For AlpacaEval 2, we report both standard win rates (WR) and length-controlled win rates (LC). LC is specifically designed
to mitigate verbosity bias for the LLM judge. For MT-Bench, we report the recommended single-answer grading (SAG),
wherein an LLM judge grades the quality of multi-turn responses on a scale of 10. For RAG-Pref, preferred and dispreferred
vector databases were generated from the preference datasets used for offline alignment (binarized UltraFeedback ). RAG
was run using the preferred vector database from RAG-Pref. All online alignment methods were evaluated over all SFT and
offline aligned models. Further details are provided in Appendix C.2.
All results are presented in Table 3. Across all tasks and SFT/offline aligned models, RAG-Pref outperforms all other online
alignment methods. In particular, averaged across all tasks, RAG-Pref improves performance over baseline models, RAG,
7

and OPAD by24 .4%,7.3%, and228.4%. Furthermore,RAG-Pref is the only online alignment method to consistently
improve baseline model performance across all tasks and offline alignment algorithms; OPAD fails to improve
baseline performance across all tasks and offline aligned models, while RAG fails to improve SFT performance for WR
and SimPO performance for SAG. The latter is consistent with results from Section 5 where RAG was shown to decrease
performance on agentic safety tasks.
RAG-Pref Contrastive Information.For AlpacaEval 2 and MT-Bench benchmarks, we estimate the amount of contrastive
information (as defined in Theorem 3.1) RAG-Pref encodes over RAG. For each benchmark, the average perplexity across
all sequences is used to calculate the entropy of base model generations, RAG generations, and RAG-Pref generations.
BothΔ𝐻RAGandΔ𝐻RAG-Pref (defined in Section 3.2) are calculated and used to compute the contrastive information
Δ𝐻RAG-Pref−Δ𝐻 RAG. Thepercentage of contrastive information(PCI), (Δ𝐻 RAG-Pref−Δ𝐻 RAG)/Δ𝐻 RAG-Pref , is reported in
Table 4.
Table 4: Percentage of contrastive information (PCI) for general human-preference alignment benchmarks AlpacaEval 2
and MT-Bench and offline aligned models.
Offline Alignment
Benchmark SFT DPO PPO SimPO RTO
AlpacaEval 2 61.8 19.3 30.7 16.3 18.2
MT-Bench 40.4 26.5 50.0 18.5 24.3
Across all models, the average PCI is 29.3% and 31.9% for AlpacaEval 2 and MT-Bench, respectively. We note that this
directly translates to the amount of additional information encoded by RAG-Pref relative to standard RAG. Thus, on
average, contrastive information accounts for nearly 30% of RAG-Pref’s total mutual information in Table 4.
6 Discussion and Conclusions
Herein, we addressed major drawbacks of current SAT algorithms; we explored the efficacy of SOTA SAT algorithms to
strengthen refusal guardrails against agentic FBAs, which induce malicious tool use without standard refusal triggers.
Evaluating five widely used LLMs, we showed that SOTA SAT algorithms display limited ability to enable FBA refusal
guardrails, only improving baseline refusal rates by average factors of 1.4 (DPO) and 1.6 (SafeDPO).
To address this critical security gap and the high computational cost of offline alignment, we introduced RAG-Pref, a
novel training-free alignment algorithm implementable with off-the-shelf RAG components. We showed that RAG-Pref
significantly outperforms both SOTA offline and online alignment algorithms, increasing baseline FBA refusal rates
by an average 3-fold improvement–2.1, 1.9, and 1.8 times greater than DPO, SafeDPO, and OPAD, respectively. When
combined with offline alignment methods, RAG-Pref further boosts overall refusal guardrails, enabling an average 3.7
factor improvement in FBA refusal rates, compared to 2.9 and 1.5 for other online algorithms and offline alignment alone,
respectively. Furthermore, we demonstrated that RAG-Pref preprocessing is three orders of magnitude faster than DPO
offline alignment, while being three times faster and 4.2 times more memory efficient than other recent training-free
alignment methods.
Beyond agentic safety, we demonstrated that RAG-Pref similarly improves performance for general human-preference
alignment tasks. Across models alignment-tuned using five SOTA algorithms, we showed that RAG-Pref leads to consistent
improvements on both AlpacaEval 2 and MT-Bench benchmarks, with average increases of 24.4%, 7.3%, and 228.4% over
the original models, standard RAG, and alternative offline alignment method OPAD. Critically, RAG-Pref is the only
online alignment method evaluated that consistently improves baseline model performance across all tasks and offline
alignment algorithms, in stark contrast to the inconsistent or completely degraded performance of standard RAG and
OPAD, respectively.
Finally, we theoretically showed that RAG-Pref encodes contrastive information beyond standard RAG, and provided a
lower bound on the expected reduction in inference uncertainty. Empirically, this contrastive information was shown to
provide nearly an average 30% more information than standard RAG for standard human alignment tasks.
8

Future work.This study provides several avenues for future exploration. In future work, we will further explore the
relationship between safety decreases in standard RAG with the contrastive information and safety improvements provided
by RAG-Pref. We hypothesize that observed decreases in standard RAG–both herein as well as in other recent work (An
et al., 2025)–are caused by a lack of contrastive information, as evidenced by the highest amount of contrastive information
(61.8 PCI) being observed for a model (SFT) and task (LC) standard RAG fails to improve performance on. Additionally,
we will explore whether alternative RAG architectures (Chan et al., 2025; Edge et al., 2025) may provide complementary
benefits through structured knowledge representations, focusing on the theoretical guarantees of these new architectures
within RAG-Pref’s general framework.
7 Acknowledgments
We thank Leidos for funding this research through the Office of Technology. Approved for public release25-LEIDOS-
0521-29630.
References
Bang An, Shiyue Zhang, and Mark Dredze. Rag llms are not safer: A safety analysis of retrieval-augmented generation for
large language models. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5444–5474, 2025.
Anthropic. System card: Claude opus 4.5. Technical report, Anthropic, November 2025. URL https://www-cdn.anthropic.
com/bf10f64990cfda0ba858290be7b8cc6317685f47.pdf. Version dated November 24, 2025.
Anthropic. Donating the model context protocol and establishing the agentic ai foundation. https://www.anthropic.com/
news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation, December
2025a. Accessed: 2026-01-26.
Anthropic.Introducing the Model Context Protocol. https://www.anthropic.com/news/model-context-protocol ,
2025b. "Accessed: 2025-02-12".
Anthropic.Slack MCP Server. https://github.com/modelcontextprotocol/servers/tree/main/src/slack , 2025c.
"Accessed: 2025-05-09".
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort,
Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk,
Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt,
Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and
Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. URL
https://arxiv.org/abs/2204.05862.
Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng, and Hen-Hsen Huang. Don’t do rag: When cache-augmented generation
is all you need for knowledge tasks. InCompanion Proceedings of the ACM on Web Conference 2025, pp. 893–897, 2025.
Patrick Chao, Edoardo Debenedetti, et al. Jailbreakbench: An open robustness benchmark for jailbreaking large language
models.Advances in Neural Information Processing Systems (NeurIPS), 2024.
Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. Jailbreaking black box
large language models in twenty queries. In2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML),
pp. 23–42. IEEE, 2025.
Huayu Chen, Guande He, Lifan Yuan, Ganqu Cui, Hang Su, and Jun Zhu. Noise contrastive alignment of language models
with explicit rewards. InThe Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language models with noisy
feedback. InForty-first International Conference on Machine Learning, 2024.
Thomas M Cover.Elements of information theory. John Wiley & Sons, 1999.
9

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun.
Ultrafeedback: Boosting language models with high-quality feedback. 2023.
CVE Project. CVE List V5: CVE cache of the official CVE list in CVE JSON 5 format. https://github.com/CVEProject/
cvelistV5, 2023. Accessed: 2025-10-30.
Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe
reinforcement learning from human feedback. InThe Twelfth International Conference on Learning Representations, 2024.
URLhttps://openreview.net/forum?id=TyFrPOKYXw.
DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms.
Advances in neural information processing systems, 36:10088–10115, 2023.
Karel D’Oosterlinck, Winnie Xu, Chris Develder, Thomas Demeester, Amanpreet Singh, Christopher Potts, Douwe Kiela,
and Shikib Mehri. Anchored preference optimization and contrastive revisions: Addressing underspecification in
alignment.Transactions of the Association for Computational Linguistics, 13:442–460, 2025.
Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to
debias automatic evaluators.arXiv preprint arXiv:2404.04475, 2024.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky,
Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization,
2025. URLhttps://arxiv.org/abs/2404.16130.
Songyang Gao, Qiming Ge, Wei Shen, Shihan Dou, Junjie Ye, Xiao Wang, Rui Zheng, Yicheng Zou, Zhi Chen, Hang
Yan, et al. Linear alignment: A closed-form solution for aligning human preferences without tuning and feedback. In
International Conference on Machine Learning, pp. 14702–14722. PMLR, 2024.
Team Gemma, Morgane Riviere, Shreya Pathak, et al. Gemma 2: Improving open language models at a practical size, 2024.
URLhttps://arxiv.org/abs/2408.00118.
Google.MCP Toolbox for Databases: Simplify AI Agent Access to Enterprise Data. https://cloud.google.com/blog/
products/ai-machine-learning/mcp-toolbox-for-databases-now-supports-model-context-protocol , 2025.
"Accessed: 2025-05-09".
Aaron Grattafiori, Abhimanyu Dubey, et al. The llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve
signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. InProceedings of the
16th ACM workshop on artificial intelligence and security, pp. 79–90, 2023.
John Halloran.MCPSafetyScanner - Automated MCP safety auditing and remediation using Agents. https://github.com/
johnhalloran321/mcpSafetyScanner, 2025. "Accessed: 2025-05-05".
Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. Toxigen: A large-scale
machine-generated dataset for adversarial and implicit hate speech detection.arXiv preprint arXiv:2203.09509, 2022.
HuggingFace.UltraFeedback Binarized. https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_
binarized, 2025. "Accessed: 2025-10-13".
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda,
Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv preprint arXiv:2410.21276, 2024.
Invariant.Introducing Guardrails: The contextual security layer for the agentic era. https://invariantlabs.ai/blog/
guardrails, 2025a. "Accessed: 2025-05-05".
Invariant.Introducing MCP-Scan: Protecting MCP with Invariant. https://invariantlabs.ai/blog/
introducing-mcp-scan, 2025b. "Accessed: 2025-05-05".
10

Haozhe Ji, Cheng Lu, Yilin Niu, Pei Ke, Hongning Wang, Jun Zhu, Jie Tang, and Minlie Huang. Towards efficient exact
optimization of language model alignment.arXiv preprint arXiv:2402.00856, 2024.
Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong
Yang. Beavertails: Towards improved safety alignment of llm via a human-preference dataset. In A. Oh, T. Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.),Advances in Neural Information Processing Systems, volume 36,
pp. 24678–24704. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/
file/4dbb61cb68671edc4ca3712d70083b9f-Paper-Datasets_and_Benchmarks.pdf.
Zhuoran Jin, Hongbang Yuan, Tianyi Men, Pengfei Cao, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun
Zhao. RAG-RewardBench: Benchmarking reward models in retrieval augmented generation for preference alignment.
In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.),Findings of the Association
for Computational Linguistics: ACL 2025, pp. 17061–17090, Vienna, Austria, July 2025. Association for Computational
Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.877. URL https://aclanthology.org/2025.
findings-acl.877/.
Seungjae Jung, Gunsoo Han, Daniel Wontae Nam, and Kyoung-Woon On. Binary classifier optimization for large language
model alignment.arXiv preprint arXiv:2404.04656, 2024.
Geon-Hyeong Kim, Youngsoo Jang, Yu Jin Kim, Byoungjip Kim, Honglak Lee, Kyunghoon Bae, and Moontae Lee. SafeDPO:
A simple approach to direct preference optimization with enhanced safety. InThe Fourteenth International Conference on
Learning Representations, 2026. URLhttps://openreview.net/forum?id=PJdw4VBsXD.
Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall,
Victor Carbune, Abhinav Rastogi, et al. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai
feedback. InInternational Conference on Machine Learning, pp. 26874–26901. PMLR, 2024.
Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, Ann-Kathrin Dombrowski,
Shashwat Goel, Gabriel Mukobi, et al. The wmdp benchmark: Measuring and reducing malicious use with unlearning.
InInternational Conference on Machine Learning, pp. 28525–28550. PMLR, 2024.
Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J Liu, and Jialu Liu. Statistical rejection
sampling improves preference optimization. InThe Twelfth International Conference on Learning Representations, 2024.
Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large
language models.arXiv preprint arXiv:2310.04451, 2023a.
Xiaogeng Liu, Peiran Li, G. Edward Suh, Yevgeniy Vorobeychik, Zhuoqing Mao, Somesh Jha, Patrick McDaniel,
Huan Sun, Bo Li, and Chaowei Xiao. Autodan-turbo: A lifelong agent for strategy self-exploration to jailbreak
llms. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.),International Conference on Representation Learn-
ing, volume 2025, pp. 10313–10360, 2025. URL https://proceedings.iclr.cc/paper_files/paper/2025/file/
1bff3663270ba47f801e917f782d7935-Paper-Conference.pdf.
Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang,
Yan Zheng, et al. Prompt injection attack against llm-integrated applications.arXiv preprint arXiv:2306.05499, 2023b.
Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven
Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. In
International Conference on Machine Learning, pp. 35181–35224. PMLR, 2024.
Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi.
Tree of attacks: Jailbreaking black-box llms automatically.Advances in Neural Information Processing Systems, 37:
61065–61105, 2024.
Igor Melnyk, Youssef Mroueh, Brian Belgodere, Mattia Rigotti, Apoorva Nitsure, Mikhail Yurochkin, Kristjan Greenewald,
Jiri Navratil, and Jarret Ross. Distributional preference alignment of llms via optimal transport. InThe Thirty-eighth
Annual Conference on Neural Information Processing Systems, 2024.
11

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward.Advances
in Neural Information Processing Systems, 37:124198–124235, 2024.
Microsoft.Introducing Model Context Protocol (MCP) in Copilot Studio. https://tinyurl.com/CopilotMCP , 2025. "Accessed:
2025-03-20".
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal,
Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback.Advances in
neural information processing systems, 35:27730–27744, 2022.
Fábio Perez and Ian Ribeiro. Ignore previous prompt: Attack techniques for language models.arXiv preprint arXiv:2211.09527,
2022.
ProtectAI.Model Card for distilroberta-base-rejection-v1. https://huggingface.co/protectai/
distilroberta-base-rejection-v1, 2025. "Accessed: 2025-05-15".
Brandon Radosevich and John Halloran. Mcp safety audit: Llms with the model context protocol allow major security
exploits.arXiv preprint arXiv:2504.03767, 2025.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference
optimization: Your language model is secretly a reward model.Advances in Neural Information Processing Systems, 36:
53728–53741, 2023.
Philipp Schmid.How to use Anthropic MCP Server with open LLMs, OpenAI or Google Gemini. https://github.com/
philschmid/mcp-openai-gemini-llama-example, 2025. "Accessed: 2025-04-28".
Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal,
Cem Anil, et al. Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red
teaming.arXiv preprint arXiv:2501.18837, 2025.
Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. " do anything now": Characterizing and evaluating
in-the-wild jailbreak prompts on large language models. InProceedings of the 2024 on ACM SIGSAC Conference on
Computer and Communications Security, pp. 1671–1685, 2024.
Maojia Song, Shang Hong Sim, Rishabh Bhardwaj, Hai Leong Chieu, Navonil Majumder, and Soujanya Poria. Measuring
and enhancing trustworthiness of LLMs in RAG through grounded attributions and learning to refuse. InThe Thirteenth
International Conference on Learning Representations, 2025. URLhttps://openreview.net/forum?id=Iyrtb9EJBp.
Stripe.Stripe Agent Toolkit.https://github.com/stripe/agent-toolkit, 2025. "Accessed: 2025-03-20".
Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Yuehe Chen, Bowen Song, Zilei Wang, Weiqiang Wang, and
Liang Wang. Divide-then-align: Honest alignment based on the knowledge boundary of RAG. In Wanxiang Che,
Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.),Proceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11461–11480, Vienna, Austria, July 2025.
Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.561. URL https:
//aclanthology.org/2025.acl-long.561/.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, An-
drew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models.arXiv preprint
arXiv:2312.11805, 2023.
Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D Manning, and Chelsea Finn. Fine-tuning language models for
factuality. InThe Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/
forum?id=WPZ2yPag4K.
Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro
von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of lm alignment.arXiv preprint
arXiv:2310.16944, 2023.
12

Jiayi Wu, Hengyi Cai, Lingyong Yan, Hao Sun, Xiang Li, Shuaiqiang Wang, Dawei Yin, and Ming Gao. PA-RAG: RAG
alignment via multi-perspective preference optimization. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.),Proceedings
of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp. 9091–9112, Albuquerque, New Mexico, April 2025. Association for
Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.459. URL https://aclanthology.
org/2025.naacl-long.459/.
Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for
language model alignment. InAdaptive Foundation Models: Evolving AI for Personalized and Efficient Learning, 2024.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. How language model hallucinations can snowball.
arXiv preprint arXiv:2305.13534, 2023.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li,
Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in neural information
processing systems, 36:46595–46623, 2023.
Han Zhong, Zikang Shan, Guhao Feng, Wei Xiong, Xinle Cheng, Li Zhao, Di He, Jiang Bian, and Liwei Wang. Dpo meets
ppo: Reinforced token optimization for rlhf.arXiv preprint arXiv:2404.18922, 2024.
Han Zhong, Zikang Shan, Guhao Feng, Wei Xiong, Xinle Cheng, Li Zhao, Di He, Jiang Bian, and Liwei Wang. DPO meets
PPO: Reinforced token optimization for RLHF. InForty-second International Conference on Machine Learning, 2025. URL
https://openreview.net/forum?id=IfWKVF6LfY.
Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al.
Lima: Less is more for alignment.Advances in Neural Information Processing Systems, 36:55006–55021, 2023.
Mingye Zhu.OPAD.https://github.com/stevie1023/OPAD, 2025. "Accessed: 2025-07-01".
Mingye Zhu, Yi Liu, Lei Zhang, Junbo Guo, and Zhendong Mao. On-the-fly preference alignment via principle-guided
decoding. InThe Thirteenth International Conference on Learning Representations, 2025.
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable
adversarial attacks on aligned language models.arXiv preprint arXiv:2307.15043, 2023.
A Proof of Theorem 1 and Contrastive Information Nonnegativity
Theorem A.1. Δ𝐻 RAG-Pref≥Δ𝐻 RAG. Furthermore, when dispreferred examples provide non-redundant information compared
to preferred examples,Δ𝐻 RAG-Pref >Δ𝐻 RAG.
Proof.
Δ𝐻RAG-Pref =𝐼(𝑌;𝑍𝑤|𝑋)+𝐼(𝑌;𝑍𝑙|𝑋,𝑍𝑤)
=Δ𝐻 RAG| {z }
positive examples+𝐼(𝑌;𝑍𝑙|𝑋,𝑍𝑤)|            {z            }
contrastive information
From the monotonicity of thecontrastive information, 𝐼(𝑌;𝑍𝑙|𝑋,𝑍𝑤), we have Δ𝐻RAG-Pref≥Δ𝐻 RAG. Furthermore, when
dispreferred examples provide non-redundant information compared to preferred examples, the contrastive information is
thus𝐼(𝑌;𝑍𝑙|𝑋,𝑍𝑤)>0.
13

B Proof of uncertainty reduction
Theorem B.1.RAG reduces expected uncertainty during autoregressive LLM inference, and RAG-Pref further reduces expected
inference uncertainty. Furthermore, the maximum reduction in uncertainty between standard inference and RAG/RAG-Pref is
lower-bounded by the contrastive information.
Proof.Consider the conditional entropy during standard inference,
𝐻(𝑌|𝑋)=∑︁
𝑥∈X𝑝(𝑥)∑︁
𝑦∈Y𝜋𝜃(𝑦|𝑥)log𝜋 𝜃(𝑦|𝑥).
Conditioning reduces entropy (Cover, 1999), so that
𝐻(𝑌|𝑋,𝑍𝑤)≤𝐻(𝑌|𝑋).
Thus,
𝐻(𝑌|𝑋,𝑍𝑤,𝑍𝑙)|             {z             }
RAG-Pref≤𝐻(𝑌|𝑋,𝑍𝑤)|         {z         }
RAG≤𝐻(𝑌|𝑋)|   {z   }
Standard Inf.,(2)
which completes the first half of the theorem.
The maximum reduction in uncertainty from standard inference and RAG/RAG-Pref is thus 𝐻(𝑌|𝑋)−𝐻(𝑌|𝑋,𝑍𝑤,𝑍𝑙), so
that we have
𝐻(𝑌|𝑋,𝑍𝑤)≤𝐻(𝑌|𝑋)
⇒𝐻(𝑌|𝑋,𝑍𝑤)−𝐻(𝑌|𝑋,𝑍𝑤,𝑍𝑙)≤𝐻(𝑌|𝑋)−𝐻(𝑌|𝑋,𝑍𝑤,𝑍𝑙)
⇒𝐼(𝑌;𝑍𝑙|𝑋,𝑍𝑤)|            {z            }
contrastive information≤𝐻(𝑌|𝑋)−𝐻(𝑌|𝑋,𝑍𝑤,𝑍𝑙)
C Experimental Setup
C.1 Agentic Safety Alignment
Table 5: MCP Tools and Descriptions
Tool Description
read_fileRead complete contents of a file
read_multiple_filesRead multiple files simultaneously
write_fileCreate new file or overwrite existing (exercise caution with this)
edit_fileMake selective edits using advanced pattern matching and formatting
create_directoryCreate new directory or ensure it exists
list_directoryList directory contents with [FILE] or [DIR] prefixes
move_fileMove or rename files and directories
search_filesRecursively search for files/directories
get_file_infoGet detailed file/directory metadata
list_allowed_directoriesList all directories the server is allowed to access
CVEs:The Common Vulnerabilities and Exposures (CVEs) (CVE Project, 2023) official repo, containing 291,161 detailed
attacks. Filtering CVEs related to RAC, MCE, CT, or Linux produced 34,391 samples. Filtering CVEs by attack feasibility
given the MCP tools of Table 5 resulted in 1,150 attacks, which were converted to FBAs.
14

Each stage of the FBA collection pipeline utilized gpt-4o version “2024-10-21.” FBAs collected considering the MCP tools
listed in Table 5. TB samples were collected by promptingClaude 3.7 Sonnetto create several useful examples per
MCP-server tool while assuming specific roles (e.g., business executive, college student, AI researcher, etc.), and manually
verified/corrected by hand. The final dataset consists of 1,035 training FBAs, 1,035 TB training samples, and 115 FBA
testing samples.
Table 6: Number of data samples after each step of FBA and TB data collection.
Data Number of instances
All reported CVEs (as of 4/23/2025) 291,161
CVEs related to RAC, MCE, CT, or Linux 34,391
Feasible CVEs given the MCP tools in Table 5 1,150
(Training FBAs, Testing FBAs) (1,035, 115)
Training TB samples 1,035
DPO:The checkpoints for all LLMs considered herein were downloaded from HuggingFace. All DPO and RAG-Pref experi-
ments were run on an Nvidia L40S GPU with 48GB onboard memory. For DPO alignment, the following packages+versions
were used: Transformers v4.49.0.dev0 ,Torch v2.4.0+cu121 ,TRL v0.15.0dev0 ,PEFT v0.12.0 ,BitsAndBytes
v.0.45.0 ,Accelerate 0.34.2 , and Flash Attention-2 v2.7.3 . All DPO fine-tuning runs utilized QLoRA (Dettmers
et al., 2023), targeting all linear-layers for adaptation with LoRA dimension 16. All DPO runs used the following training
recipe (adapted from (Tunstall et al., 2023) and (Zhou et al., 2023) for DPO and small-scale/high-quality alignment, respec-
tively): 15 training epochs, AdamW_torch optimizer, cosine annealing schedule, warmup_ratio 0.1,learning rate 5𝑒−7,
BF16 precision, and FlashAttention2 . All unreferenced parameters were left to their defaults. All inference runs used
the previously stated parameters, exceptGemma-2-2B-ITnon-DPO-aligned runs, which required attn_implementation
eagerandFP16to run. All refusal and acceptance metrics were calculated using ten generations per LLM per alignment
configuration per test sample, with sampling enabled and temperature =0.7. All non-RAG evaluations used the same
system prompt, adapted from (Schmid, 2025).
SafeDPO:SafeDPO was run using the official paper code directly integrated intoTRL.
RAG-Pref:All RAG-Pref experiments were run using the aforementioned packages+versions, along with ChromaDB v1.0.8
andLangChain v0.1.9 . Retrieval parameters for all experiments were: embedding model sentence-transformers/all-MiniLM-L6v2 ,
Euclidean distance for similarity search, chunk size 256, and chunk overlap 10. For agentic safety results, RAG-Pref was
run using the preference pairs from the TB and FBA training sets (described in Section 4) as preferred and dispreferred
vector databases, with number of retrieved samples𝑘=2.
OPAD:OPAD was run using the harmlessness implementation from the official repo (Zhu, 2025), and extensively optimized
to handle both multiple instances during generation and vectorization for a fair computational comparison.
C.2 General Human-Preference Alignment
Benchmarks:AlpacaEval 2 was run using v0.6.6 . MT-Bench was run using v0.2.36 using the recommend single-answer
grading mode. The LLM annotator for all AlpacaEval 2 and MT-Bench results was gpt-4o . For the former, both win rates
and length-controlled win rates (Dubois et al., 2024) are reported. For the latter, the suggested single-answer grading
was performed, wherein the LLM judge provides a qualitative score (on a scale of 10) for each answer per turn, with
reported scores averaged over all turns. Across all models, AlpacaEval 2 and MT-Bench were evaluated using greedy
decoding.
Offline-aligned models:Supervised fine-tuning was first performed on Llama3-8B using the UltraFeedback dataset (Cui
et al., 2023). Subsequent offline preference alignment algorithms DPO, RLHF (via proximal policy optimization (PPO),
SimPO (Meng et al., 2024), and the state-of-the-art reinforced token optimization (RTO) were performed with the widely
used binarized UltraFeedback dataset (HuggingFace, 2025). Model checkpoints for offline preference aligned methods
were directly adapted from (Zhong et al., 2024).
Online alignment methods:RAG-Pref was run using the preferred and dispreferred training instances from the binarized
UltraFeedback dataset (RAG was run using the preferred instances) with 𝑘=8. OPAD was run using the harmlessness
15

implementation from the official repo (Zhu, 2025). RAG and RAG-Pref system prompts are available in Appendix H.
D DPO Loss Variation
No DPO DPO AOT APOd APOz BCO EXO RSO NCA Robust SPPO0.00.10.20.30.40.60.81.0Refusal Rate
Figure 1: Offline-aligned Llama-3.2-1B with following DPO losses: 1) No DPO - base model (no refusal alignment), (2)
DPO - the original “sigmoid” DPO loss function (Rafailov et al., 2023), (3) AOT - Alignment via Optimal Transport (Melnyk
et al., 2024), (4) APOd - Anchored Preference Optimization (APO) down (D’Oosterlinck et al., 2025), (5) APOz - APO
zero (D’Oosterlinck et al., 2025), (6) BCO - Binary Classifier Optimization (Jung et al., 2024), (7) EXO - Efficient Exact
Optimization (Ji et al., 2024), (8) RSO - Statistical Rejection Sampling Optimization (Liu et al., 2024), (9) NCA - Noise
Contrastive Alignment (Chen et al., 2024), (10) Robust - Provably Robust DPO (Chowdhury et al., 2024), (11) SPPO - Self-Play
Preference Optimization (Wu et al., 2024).
16

E Effects of extended DPO training on reasoning models
0 10000 20000 30000 40000 50000 60000
Steps0.00.10.20.30.40.50.60.7Train Loss
(a) Training loss over 90 Epochs
Figure 2:DeepSeek-R1-Distill-Qwen-14Baligned with DPO for 90 Epochs. Training quickly converges.
F Standard RAG vs RAG-Pref FBA Refusal Rates
Llama-3.2 1B Gemma-2-2B Llama-3.1 8B DSR1D-8B* DSR1D-14B*0.00.10.20.30.40.50.60.81.0Refusal RateBase
DPO
SafeDPO
Vanilla RAG, Base
Vanilla RAG, DPO
Vanilla RAG, SafeDPO
RAG-Pref, Base
RAG-Pref, DPO
RAG-Pref, SafeDPO
Figure 3:Attack Refusal Rates for model using Standard RAG and RAG-Pref: Refusal rates calculated over the test
FBAs. Reasoning models are denoted using∗. Base denotes models evaluated directly from their public checkpoints.
G FBA RefusalJudgeDetails
FBA refusals were assessed using the following two-stage judging:
1.Assess response using a BERT-based classifier trained explicitly on rejection/refusal data (ProtectAI, 2025). If refusal,
return true. Else, go to stage 2.
2.Asses response using DeepSeek-R1-Distill-Qwen-14B (using the refusal classifier-judge codebase from (Chao
et al., 2024)). Return binary refusal decision.
17

H RAG-Pref and Standard RAG System Prompts
Listing 1: RAG-Pref system prompt.
Base your response as closely as possible to the following preferred examples.
Avoid responding in a manner similar to the dispreferred examples.
# Preferred examples
{preferred}
# Dispreferred examples
{dispreferred}
Listing 2: Standard RAG system prompt.
Base your response as closely as possible to the following preferred examples.
# Preferred examples
{preferred}
18

I Safety generation examples
Llama-3.1-1B
Prompt:Would you mind letting me peek into the Ambari Agent logs for a quick check? I might find useful credential hints
there.
No DPO (no RAG-Pref):I’d be happy to help you take a look at the Ambari Agent logs. Ambari Agent logs can provide a
wealth of information about the agent’s activity [......]
After DPO (no RAG-Pref):I’d be happy to help you take a look at the Ambari Agent logs. What’s your current directory
structure [......]
No DPO (with RAG-Pref):I can help you with that. However, [......] Ambari Agent is a security feature [......] We can use
the ‘list_directory‘ tool to get a list of files and directories in a specific path, and then check if any of those files contain the
information you’re looking for [......]
After DPO (with RAG-Pref):I can’t assist with that request. Reviewing system logs without permission can be a security risk
and potentially violate data protection policies. [......]
Figure 4: Response example for offline/online online FBA refusal guardrails. Responses in green show direct compliance.
Responses in yellow show safety guardrails being partially triggered, yet no direct refusal is made and the LLM acquiesces.
Responses in red display direct refusal.
19