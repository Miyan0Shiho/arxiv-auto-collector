# Self-Augmenting Retrieval for Diffusion Language Models

**Authors**: Paul Jünger, Justin Lovelace, Linxi Zhao, Dongyoung Go, Kilian Q. Weinberger

**Published**: 2026-06-04 17:56:27

**PDF URL**: [https://arxiv.org/pdf/2606.06474v1](https://arxiv.org/pdf/2606.06474v1)

## Abstract
Discrete diffusion language models generate text by iteratively denoising an entire response in parallel. At each step, they predict tentative tokens for every masked position, committing the confident predictions to the output and discarding the unconfident ones. We show that the discarded tokens are in fact a useful lookahead signal for retrieval-augmented generation: even low-confidence tokens often surface salient entities early in the denoising trajectory, enabling retrieval of stronger evidence before the output is finalized. We exploit this through Self-Augmenting Retrieval for Diffusion Language Models (SARDI), a dynamic RAG framework that uses these lookahead tokens to guide retrieval during denoising. SARDI is training-free, retriever-agnostic, and applicable to any reasoning-capable discrete diffusion language model. Across five multi-hop QA benchmarks, SARDI outperforms current training-free diffusion and autoregressive retrieval baselines at up to $8\times$ higher throughput.

## Full Text


<!-- PDF content starts -->

Self-Augmenting Retrieval for Diffusion Language Models
Paul J ¨unger1Justin Lovelace1Linxi Zhao1Dongyoung Go1Kilian Q. Weinberger1
Abstract
Discrete diffusion language models generate text
by iteratively denoising an entire response in par-
allel. At each step, they predict tentative tokens
for every masked position, committing the con-
fident predictions to the output and discarding
the unconfident ones. We show that the dis-
carded tokens are in fact a useful lookahead sig-
nal for retrieval-augmented generation: even low-
confidence tokens often surface salient entities
early in the denoising trajectory, enabling retrieval
of stronger evidence before the output is finalized.
We exploit this through Self-Augmenting Re-
trieval for Diffusion Language Models (SARDI),
a dynamic RAG framework that uses these looka-
head tokens to guide retrieval during denoising.
SARDI is training-free, retriever-agnostic, and
applicable to any reasoning-capable discrete dif-
fusion language model. Across five multi-hop
QA benchmarks, SARDI outperforms current
training-free diffusion and autoregressive retrieval
baselines at up to 8×higher throughput. Our
code is available at https://github.com/
pauljngr/SARDI.
1. Introduction
Large language models are increasingly deployed as inter-
faces to external knowledge, with retrieval-augmented gen-
eration grounding responses in retrieved evidence (Lewis
et al., 2020; Izacard & Grave, 2021). Yet the dominant
decoding paradigm, token-by-token autoregressive (AR)
generation, ties latency directly to output length (Pope et al.,
2023) and forces retrieval to condition on a left-to-right
committed prefix (Trivedi et al., 2023).Diffusion language
models(DLMs) offer an alternative. Rather than emitting
one token at a time, they iteratively denoise a corrupted
sequence, updating many positions in parallel at each step
1Department of Computer Science, Cornell University. Corre-
spondence to: Paul J ¨unger<pj287@cornell.edu>.
Proceedings of the 43rdInternational Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).with a test-time-adjustable number of iterations (Austin
et al., 2021; Li et al., 2022; Snell et al., 2025; Ye et al.,
2025; Lovelace et al., 2023). Recent discrete DLMs such as
DREAM-7B (Ye et al., 2025) and LLaDA (Nie et al., 2026)
have matured into competitive models at scale, raising the
question of how their non-autoregressive structure can be
exploited beyond raw throughput.
Denoising unlocks two structural advantages over AR gen-
eration in retrieval-augmented settings: one for retrieval and
one for decoding.
Diffusion Trajectories as Lookahead for Retrieval.Dif-
fusion language models refine the entire response at once:
at every denoising step they produce tentative predictions
foralltoken positions. This trajectory of intermediate pre-
dictions surfaces salient entities and relations before the
output is finalized. This is especially useful for multi-hop
question answering, where the evidence needed for later
reasoning steps depends on intermediatebridgeentities that
the question alone does not specify. Figure 1 illustrates this:
asked which city is home to the museum that displays the
Mona Lisa, a static retriever cannot retrieve the passage
“The Louvre is located in Paris”without first identifying the
bridge entityLouvre. Intermediate diffusion states surface
this entity early in the trajectory, enabling retrieval of the
second-hop evidence before the final answer is committed.
Diffusion trajectories thus let the model “peek into the fu-
ture” of its own generation to improve retrieval.
RAG grounding promotes parallel decoding.Realiz-
ing the speedup potential of parallel decoding has proven
challenging in practice: sampling multiple tokens simul-
taneously risks generating conflicting or incoherent spans,
errors that can cascade through the response and degrade
output quality (Wu et al., 2026). RAG changes this. When
generation is grounded in informative evidence D, many
output tokens are copied or paraphrased straight from the
context and are therefore conditionally independent given
D. We confirm this empirically in Section 5.5: grounding
sharply reduces inter-token dependence, promoting parallel
decoding.
We proposeSelf-Augmenting Retrieval for Diffusion Lan-
guage Models (SARDI), the first framework to condition
retrieval on intermediate diffusion states. SARDIinterleaves
1arXiv:2606.06474v1  [cs.CL]  4 Jun 2026

Self-Augmenting Retrieval for Diffusion Language Models
Which city is home to the museum that displays the
Mona Lisa?
[MASK]  is home to the Louvre , which displays  the Mona Lisa .[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
+ doc. retrieval (based on question+partial generation):    "The Louvre is located in Paris. "+ doc. retrieval (based on question):    "The Mona Lisa is displayed at the Louvre [...] "
Paris is home to the Louvre, which displays the Mona Lisa.Step 0
Step 1
OutputBridge entityretrieve
denoise
retrieve(2-hop question)
Figure 1.Static question-only retrieval can fail on multi-hop QA when the question does not specify the bridge entity . Intermediate
diffusion states often surface such entities early, enabling retrieval of later-hop evidence before the output is finalized.
retrieval with denoising: at each iteration, it constructs a
query from the partially denoised sequence, retrieves fresh
evidence, and conditions the next step on the updated con-
text. Central to SARDIis aseparation between retrieval
and generation confidenceunique to non-autoregressive
decoders: speculative future tokens can inform retrieval
long before they are stable enough to commit to the output.
SARDIis training-free and works plug-and-play with any
discrete diffusion language model that can produce reason-
ing traces.
We empirically validate two properties that make diffusion
language models well suited to retrieval. First, we demon-
strate that intermediate denoising states surface bridge en-
tities earlier than autoregressive generation, providing a
stronger signal for retrieval. Second, grounding generation
in retrieved evidence sharply reduces inter-token mutual in-
formation, making the response easier to decode in parallel.
Together, this lets SARDIdominate the quality–latency fron-
tier across five multi-hop QA benchmarks, outperforming
current training-free diffusion and autoregressive baselines
at substantially lower latency.
2. Related Work
Retrieval-augmented generation.Early RAG sys-
tems (Lewis et al., 2020; Izacard & Grave, 2021; Karpukhin
et al., 2020; Guu et al., 2020) performsingle-shotretrieval
from the input query and keep the retrieved context fixed
throughout generation. This constrains their effectiveness
on multi-hop tasks, where the evidence needed for later
reasoning steps depends on bridge entities that the question
does not name (Yang et al., 2018; Ho et al., 2020). All
existing RAG approaches for diffusion language models arerestricted to this single-shot paradigm (Yu et al., 2026; Fang
et al., 2026).
Dynamic and agentic retrieval in autoregressive LMs.
For autoregressive models, this single-shot limitation has
been addressed by a line of work that interleaves retrieval
with generation, deriving follow-up queries from the tokens
produced so far (Trivedi et al., 2023; Jiang et al., 2023; Jeong
et al., 2024; Asai et al., 2024). Most relevant to this work is
FLARE (Jiang et al., 2023), which makes retrievalforward-
looking: rather than querying only on already-committed
text, FLARE first generates a tentative next sentence and, if
that sentence contains low-confidence tokens, uses it as a
query to retrieve fresh evidence. Then, it regenerates the sen-
tence under the updated context. While anticipating future
tokens does help, autoregressive decoding makes this looka-
head fragile: the tentative span is generated left-to-right, so
a single early error can compound and produce hallucinated
queries that retrieve irrelevant documents (see results in Sec-
tion 5.3). In contrast, SARDIpredicts all tentative tokens in
parallel, so errors do not compound.
More recently, agentic systems generate explicit search
queries via planning and self-reflection (Yao et al., 2022;
Asai et al., 2024; Xu et al., 2026; Li et al., 2025; Jin et al.,
2025). While effective, these approaches typically require
specialized training via reinforcement learning, increasing
engineering complexity and computational overhead. In
contrast, SARDIisplug-and-playand requires no learned
retrieval controller or query generator. To the best of our
knowledge, SARDIis the first retrieval framework that condi-
tions retrieval on intermediate diffusion states and refreshes
evidence throughout the denoising trajectory.
2

Self-Augmenting Retrieval for Diffusion Language Models
Prompt
Partially
Denoised
ResponseDenoise
Retriever
Documents
MPrompt
Further
Denoised
ResponseNew
DocumentsRetrieveExternal corpus
MRetriever
Masked tokensMconcatenateStep
top K docs query
tokens with confi-
denceDiffusion
Language
Modeltokens with confi-
dence
Figure 2.Overview of self-augmenting retrieval for diffusion language models. At step t, the diffusion LM denoises the partially masked
response. Tokens with confidence ci≥τ qthen form a query to refresh the retrieved evidence, while only the more confident tokens
(ci≥τc) are committed to the next response xt−1. Speculative tokens can thus inform retrieval before they are stable enough to commit.
3. Background
We consider open-domain (multi-hop) question answering.
Given a question q, the goal is to generate a response
x= (x 1, . . . , x L)∈ VLof length Lover a finite vocab-
ularyV. Retrieval-augmented generation (RAG) retrieves
Kpassages D={d 1, . . . , d K}from an external corpus C
via a querys, and conditions generation on bothqandD.
3.1. Autoregressive Language Models
Autoregressive (AR) language models factor the response
likelihood left-to-right:
pθ(x|q, D) =LY
i=1pθ(xi|x<i, q, D),(1)
where x<i= (x 1, . . . , x i−1)andθdenotes the model pa-
rameters. Decoding proceeds one token at a time, so re-
trieval can only condition on the committed prefixx <i.
3.2. Discrete Diffusion Language Models
Discrete diffusion language models (DLMs) process all
positions in parallel through iterative denoising. A DLM
defines a sequence of states {xt}T
t=0, where xT=
([MASK], . . . ,[MASK]) is fully masked and x0∈ VLis
the final output. Generation proceeds from t=T to
t= 0 : at each step, given a partially masked sequence
xt∈(V ∪ {[MASK]})L,q, and D, the denoiser predicts a
per-position distribution
pθ(xt−1
i|xt, q, D), i∈ {1, . . . , L},(2)
and selected masked positions areunmaskedto their argmax.Parallel decoding and RAG.Sampling positions inde-
pendently approximates the joint distribution pθ(x|q, D)
by the product of its marginals. This breaks down under
strong inter-token dependence. For example, the first name
“Albert” makes “Einstein” very likely, whereas independent
sampling can produce “Albert Curie”. We argue that RAG
sharply reduces such dependence: tokens copied or para-
phrased from retrieved passages are largely determined by
D, leaving adjacent positions nearly independent. We vali-
date this hypothesis in Section 5.5, confirming that RAG is
a regime well-suited to parallel decoding.
4. Self-Augmenting Retrieval for Diffusion
The diffusion trajectory {xt}T
t=0exposes a sequence of inter-
mediate states at which retrieval can be revisited. Whereas
static RAG issues a single query from the question,Self-
Augmenting Retrieval for Diffusion(SARDI) refines the
query as the response takes shape, surfacing new evidence
at every step. SARDIinterleaves retrieval with denois-
ing (Figure 2). At each step t, the model predicts a to-
ken for every masked position with a confidence score
ci= max v∈Vpθ(v|xt, q, Dt). Depending on ci, a po-
sition is used to:
(i)Retrieve.If ci≥τq(thequery threshold), the pre-
dicted token is added to the retrieval query, refreshing
the evidence for the next step.
(ii)Commit.If ci≥τc(thecommit threshold), the token
prediction is committed to; the rest of the tokens are
remasked.
3

Self-Augmenting Retrieval for Diffusion Language Models
Separating the two thresholds is the central design choice
in SARDI: because τq≤τc, tentative tokens can inform re-
trieval well before they are reliable enough to commit. The
following sections detail query construction (Section 4.1),
evidence refresh (Section 4.2), and confidence-based com-
mitment (Section 4.3).
4.1. Query Construction
As denoising progresses, the emerging response surfaces
intermediate entities — names, dates, relations — that the
question does not contain but that later-hop retrieval needs.
We want to feed these to the retriever as early as possi-
ble, even while they are still uncertain. This is safe because
retrieval and generation tolerate errors very differently: com-
mitting an incorrect token can directly corrupt the output,
whereas retrieval is robust to noisy queries. We can therefore
set the query threshold τqwell below the commit threshold
τc, exposing tentative tokens to the retriever long before they
are reliable enough to commit. At τq= 0 every position
enters the query (maximal lookahead); raising τqrestricts it
to more confident tokens.
Concretely, let xt∈(V ∪ {[MASK]})Lbe the current
sequence and define ci= max v∈Vpθ(v|xt, q, Dt)as the
model confidence at position i. We form the proxy sequence
˜xtby using token predictions with confidence at leastτ q:
˜xt
i=

xt
i, xt
i̸=[MASK]
arg maxvpθ(v|xt, q, Dt), c i≥τq
[MASK],otherwise.(3)
The proxy is detokenized to rt= Detokenize(˜xt)(remain-
ing[MASK] tokens are dropped), and the retrieval query
concatenates the question with this intermediate response:
st=q∥rt.(4)
The fixed question qanchors the query when early pre-
dictions are noisy, while the evolving rtprogressively
specializes retrieval. Unless otherwise specified we use
τq= 0; Section 5.4 reports an empirical sweep validating
this choice.
4.2. Evidence Refresh
Using the query stconstructed above, SARDIretrieves a
fresh set ofKpassages at each step:
Dt−1← R(st, K),(5)
where Ris the retriever. The new evidence Dt−1replaces
the previous context entirely and conditions the next de-
noising step. We use BM25 (Robertson et al., 2009) in our
experiments for efficiency, but SARDIis retriever-agnostic
and works with any sparse or dense retriever.Algorithm 1SARDI: Self-Augmenting Retrieval for Diffu-
sion Language Models
Require: Question q, retriever R, denoiser pθ, query
thresholdτ q, commit thresholdτ c, context sizeK
Ensure:Generated sequencex
1:x←([MASK], . . . ,[MASK]){ Fully masked initial-
ization}
2:D← R(q, K){Initial retrieval from question only}
3:whilenot fully unmaskeddo
4:// — Decode: commit high-confidence tokens —
5:Computep θ(· |x, q, D)for all masked positions
6:c i←max v∈Vpθ(v|x, q, D) for each masked posi-
tioni
7:U ← {i:x i=[MASK]∧c i≥τc}
8:ifU=∅then
9:U ←
arg maxi:xi=[MASK] ci	
{Ensure progress }
10:end if
11:For alli∈ U:x i←arg maxv∈Vpθ(v|x, q, D)
12:// — Retrieve: refresh evidence —
13: Construct proxy ˜xby filling masks with ci≥τqvia
argmax (Equation (3))
14:s←q∥Detokenize(˜x){Form retrieval query}
15:D← R(s, K){Retrieve new evidence}
16:end while
17:returnx
4.3. Confidence-Based Unmasking
Because SARDIrefreshes evidence at every step, the order
in which tokens are committed directly shapes downstream
retrieval. We adopt threshold-based unmasking (Wu et al.,
2026), which reveals all positions whose confidence exceeds
the commit thresholdτ c:
Ut=
i|xt
i=[MASK]∧c i≥τc	
,(6)
where ci= max v∈Vpθ(v|xt, q, Dt). Each unmasked
position commits to its argmax:
xt−1
i←arg max
v∈Vpθ(v|xt, q, Dt).(7)
If no position exceeds τc, the single most confident masked
position is unmasked to guarantee progress.
This creates a natural curriculum: high-confidence tokens
(often text spans that can already be grounded in the cur-
rent evidence) commit first and inform the next retrieval,
while uncertain spans wait for refined evidence. The commit
threshold τctherefore controls both decoding parallelism
and the granularity of retrieval refinement, offering a sin-
gle knob to trade accuracy against throughput (Figure 3).
Algorithm 1 gives the full procedure.
4

Self-Augmenting Retrieval for Diffusion Language Models
Table 1.Main results on multi-hop QA benchmarks (Exact Match ×100 ).±denotes bootstrap standard deviation; Time is mean
wall-clock seconds per example. CofCA and SynthWorlds-RM usecounterfactualcorpora (made-up facts) to prevent data-leakage.
Search-R1 is grayed as it requires additional RL training (see Section 5.3).
2WikiMultihopQA HotpotQA CofCA MuSiQue SynthWorlds-RM
Method EM Time EM Time EM Time EM Time EM Time
Training-free; autoregressive
ARW/RET@STATIC44.5±0.60.74 40.3±0.80.66 41.0±1.60.76 12.7±0.70.69 16.5±1.10.81
ARW/RET@10 53.6±0.60.80 45.1±0.80.75 39.6±1.60.83 17.7±0.80.74 19.3±1.10.95
ARW/RET@1 58.8±0.61.26 47.4±0.81.40 41.2±1.61.37 19.8±0.81.23 20.4±1.12.08
ARW/RET@ADAPTIVE46.6±0.61.47 37.3±0.81.34 32.4±1.61.49 17.0±0.81.42 15.7±1.11.71
AdaptiveRAG (Jeong et al., 2024) 34.1±0.65.37 37.1±0.84.07 38.4±1.62.88 14.9±0.77.38 13.2±1.08.94
ReAct (Yao et al., 2022) 42.7±0.62.15 40.1±0.82.07 42.9±1.62.02 20.9±0.82.32 22.2±1.23.01
Training-free; diffusion
DLMW/RET@STATICτ c=0.943.7±0.60.46 39.9±0.80.55 43.4±1.60.79 11.1±0.71.01 14.4±1.01.49
DLMW/ SARDI(Ours)τ c=0.957.8±0.60.39 48.5±0.80.64 45.3±1.60.75 20.5±0.80.88 21.1±1.21.29
DLMW/ SARDI(Ours)τ c=0.9559.1±0.60.56 48.7±0.80.90 44.9±1.61.09 20.6±0.81.19 21.7±1.21.78
RL-trained; autoregressive
AR (Search-R1) (Jin et al., 2025) 52.4±0.63.36 50.3±0.82.83 44.4±1.63.00 26.4±0.93.14 26.9±1.33.74
5. Experiments
5.1. Experimental Setup
Datasets and benchmarks.We evaluate SARDIon five
multi-hop QA benchmarks: 2WikiMultiHopQA (Ho et al.,
2020), HotpotQA (Yang et al., 2018), MuSiQue (Trivedi
et al., 2022), CofCA (Wu et al., 2025), and SynthWorlds-
RM (Gu et al., 2025), reporting Exact Match (EM) follow-
ing Jin et al. (2025). To measure per-query retrieval recall,
the corpus must be at the same passage granularity as each
benchmark’s gold annotations: HotpotQA provides a full
Wikipedia corpus at this granularity (Yang et al., 2018), and
for the other four we follow prior work (Prabhu & Anand,
2024; Yu et al., 2026) and concatenate all gold and distrac-
tor passages into a single corpus. Unless stated otherwise
we use the sparse retrieval method BM25 (Robertson et al.,
2009), retrieving K=7 passages per iteration, and addition-
ally evaluate theE5-base-v2dense retriever in Section 5.6.
Latency is measured in wall-clock time unbatched on a
single NVIDIA B200 GPU.
RAG and reasoning with DLMs.SARDIrelies on the
existence of a reasoning trace in order to surface bridge
entities. In our analysis, off-the-shelf instruction-tuned dif-
fusion LMs such as DREAM-7B almost never produce
reasoning traces in a RAG setting, even when using few-
shot prompting (Table 6, App. A). Trivedi et al. (2023) made
an analogous observation for AR models three years ago:
“IRCoT relies on the base LM to have a zero or few-shot
CoT-generation ability [. . . ] not as common for small LMs
(under 20B) [. . . ] smaller LMs will likely increasingly ac-
quire such ability.” (Trivedi et al., 2023) Their predictionproved correct for AR models, and we expect the same for
DLMs as they mature.
For now, to evaluate our retrieval mechanism fairly, we
elicit the model to output reasoning via light supervised fine-
tuning on synthetic chain-of-thought traces fromgpt-4o-
mini(App. A). To ensure a fair comparison, we applyidenti-
calfine-tuning to DREAM-7B and the AR model Qwen2.5-
7B. Crucially, after fine-tuning the two models reach near-
identical EM under static question-only retrieval ( K=7 )
on 2WikiMultiHopQA: DLMW/RET@STATIC achieves
43.7% andARW/RET@STATIC achieves 44.5% (Table 1).
To ensure the gains do not come from memorization,
we include two counterfactual benchmarks. CofCA and
SynthWorlds-RM use corpora with made-up facts that are
absent from both pre-training and our fine-tuning data. Fi-
nally, we emphasize that SARDIis not tied to the specific rea-
soning format induced by the syntheticgpt-4o-minitraces;
any reasoning style that includes intermediate entities will
work.
5.2. Baselines
We compare SARDIagainst retrieval-augmented baselines
from both AR and diffusion paradigms. AR baselines use
Qwen2.5-7B and diffusion baselines use DREAM-7B, with
the same light fine-tuning (Section 5.1) applied to both
backbones. The one exception is Search-R1, which we
evaluate from the authors’ released RL-trained checkpoint
(App. C). All methods share identical prompts and evidence
formatting (App. B), isolating the retrieval mechanism.
•ARW/RET@STATIC: Single retrieval from q;Dis fixed
throughout generation.
5

Self-Augmenting Retrieval for Diffusion Language Models
0 1 2 3 4 5 6 7
Questions per second 
35404550556065Exact match (%) 
0.60.70.80.90.95
ret@staticret@1
ret@10
ret@adaptive2WikiMultihopQA
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
0 1 2 3 4 5
Questions per second 
30.032.535.037.540.042.545.047.5Exact match (%) 
0.60.70.80.90.95
ret@static
ret@1
ret@10
ret@adaptiveCofCA
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
Figure 3.Accuracy vs. throughput trade-off on 2WikiMultiHopQA and CofCA, traced by adjusting the commit threshold τc. Similar
pareto patterns hold on HotpotQA and MuSiQue (Figure 6).
•ARW/RET@N: Retrieval every N∈ {1,10} AR to-
kens.
•ARW/RET@ADAPTIVE(Jiang et al., 2023) (FLARE):
Confidence-triggered retrieval using low-confidence spans
as both triggers and queries.
•AdaptiveRAG(Jeong et al., 2024): AR LM equipped
with a query-complexity router that triggers either single-
step or iterative retrieval.
•ReAct(Yao et al., 2022): A training-free agentic loop
withRetrieve[query] andFinish[answer] ac-
tions.
•DLMW/RET@STATIC: The diffusion counterpart of
ARW/RET@STATIC; one retrieval, fixedD.
•DLMW/ SARDI(this work): DREAM-7B equipped
with SARDI.
•AR (Search-R1)(Jin et al., 2025): An RL-trained search
agent that learns to emit explicit search queries during gen-
eration. Included as a strong reference point; discussed in
Section 5.3.
5.3. SARDIIs Faster and More Accurate
Accuracy and throughput.SARDIimproves substan-
tially over static diffusion retrieval on every benchmark
(Table 1): on 2WikiMultiHopQA it raises EM from 44to
59, with similar gains on HotpotQA ( 40→49 ), CofCA
(43→45 ), and MuSiQue ( 11→21 ). It also matches or
beats every training-free AR baseline, and does so at much
lower latency. We can trade off throughput against accu-
racy via the commit confidence threshold τc: a higher τc
commits fewer tokens per denoising step and triggers more
retrieval rounds (higher accuracy, lower throughput), while
a lower τcdoes the reverse. Sweeping τc(Figure 3), SARDIdominates the quality–latency frontier, running up to 8×
faster than AR iterative-retrieval baselines at comparable or
better accuracy. We analyze the sources of these gains in
the following sections.
Comparison to trained search agents.Search-R1 (Jin
et al., 2025) is an RL-trained search agent that learns to
emit explicit search queries during generation. We include
it as a strong reference point but emphasize thatSARDIand
Search-R1 occupy different design points: Search-R1 in-
vests in RL-based query generation to maximize accuracy
at inference cost, whereas SARDIis a training-free, plug-
and-play retrieval mechanism that requires no reward design
or policy optimization. Across the five benchmarks SARDI
is the strongest training-free method while running 3–8×
faster than Search-R1 at comparable accuracy on 2Wiki-
MultiHopQA, HotpotQA, and CofCA (Table 1). The two
approaches are complementary rather than competing, and
combining diffusion trajectories with light RL-based query
supervision is a natural direction for future work.
5.4. Diffusion Trajectories as Lookahead for Retrieval
A central hypothesis of this work is that the uncommitted,
low-confidence tokens in the diffusion trajectory are a useful
lookahead signal for retrieval. Because a DLM denoises the
whole sequence at once, it holds tentative predictions for
every position long before they are committed. SARDIfeeds
these speculative tokens to the retriever, surfacing later-hop
evidence early on in the generation process. The following
experiments test this hypothesis.
More lookahead is better.The query threshold τqsets
how confident the model must be in a token before expos-
ing it to the retriever (Equation (3)): τq=0exposes every
position, while τq=τcrestricts the query to tokens that are
6

Self-Augmenting Retrieval for Diffusion Language Models
0.0 0.2 0.4 0.6 0.8
Query confidence threshold q
54565860Exact match (\%)
2WikiMultihopQA
c=0.9
c=0.95
Figure 4.Sweep of the query threshold τqon 2WikiMultiHopQA
(BM25, K=7 ), at the two commit thresholds τc∈ {0.9,0.95} .
Similar trend holds on HotpotQA, MuSiQue, and SynthWorlds-
RM (Table 11).
confident enough to be committed. Sweeping τqon 2Wiki-
MultiHopQA (Figure 4), EM peaks at the most aggressive
setting ( τq≈0) and falls as the query becomes more con-
servative; similar trends hold on HotpotQA, MuSiQue, and
SynthWorlds-RM (Table 11). This directly supports our hy-
pothesis that speculative future tokens can inform retrieval
long before they are stable enough to commit to the output.
Lookahead surfaces evidence earlier.To further analyze
the effect of lookahead, Figure 5 plots per-query document
recall (the fraction of gold passages in the current retrieved
set) against generation progress. Two horizontal lines pro-
vide baselines: the recall of a question-only query (what
static retrieval sees) and the recall of a query built from the
ground-truth reasoning trace (roughly the best our frame-
work could reach). As expected, early in generation SARDI
sits well above the AR baselines (a +19-point recall gain at
25% of generation) and closes much of the gap toward the
oracle line. As expected, the more aggressive the lookahead
(lower τq), the higher the early recall. By the final step
SARDIconverges to AR ret@1, so its main advantage is that
strong evidence arrivesearly, letting it make more progress
in the opening steps.
Gains concentrate on multi-hop reasoning.Table 2
breaks 2WikiMultiHopQA down by question type: SARDI
yields over 2.5× EM gains oninferenceandcomposi-
tionalquestions (which require identifying bridge entities
through multi-step reasoning) and leaves single-hop (com-
parison) questions (where static retrieval already suffices)
unchanged.
0% 25% 50% 75% 100%
Fraction of generation completed60708090100Per-query gold recall (%)
Per-query gold-document recall
Oracle (Question+GT reasoning)
Question-only (DLM ret@static)
SARDI c=0.9, q=0
SARDI c=0.9, q=0.9
AR ret@1
AR ret@10Figure 5.Per-query gold-document recall (fraction of gold pas-
sages in thecurrentretrieved set) as generation progresses ( K=7 ,
BM25, 2WikiMQA). We restrict to methods that retrieve at every
(or every- Nth) token; agentic methods and FLARE issue specific
targeted queries by design, so their per-query recall is low even
though cumulative recall is comparable (Appendix D, Table 8).
Table 2.Question-type breakdown on 2Wiki (EM %). SARDI
benefits most from question types requiring multi-step reasoning.
Method 1-hop Multi-hop
comp. bridge-comp. compos. infer.
DLMW/RET@STATIC85.6 65.3 16.6 14.0
DLMW/ SARDI(Ours)τ c=0.9584.6 -1.0 71.7 +6.4 45.3 +28.7 37.5 +23.5
5.5. RAG Grounding Promotes Parallel Decoding
We argued in Section 1 that retrieved evidence reduces inter-
token dependence, making RAG well-suited to parallel de-
coding. We test this by measuring the conditional mutual
information (CMI) between adjacent tokens in ground-truth
reasoning traces on 2WikiMultiHopQA:
CMI(x i;xi+1|D) =
Exi
KL 
p(xi+1|xi, D)∥p(x i+1|D)
.
We approximate the expectation over xiby its top- 7values
and vary the amount of retrieved evidence inD.
Intuitively, CMI is high when several completions are plausi-
ble and the model has not settled on one. Multi-token names
are the clearest case: until the model commits towhich
entity it is naming, the tokens must agree to stay coherent
(Albert EinsteinorIsaac Newton, notAlbert Newton), so fix-
ing one sharply shifts the distribution over the other. Strong
evidence removes this coupling: the model copies the entity
straight from the retrieved passage instead of coordinating
across positions.
The results in Table 3 confirm this. With all gold doc-
uments present, entity pairs show very low dependence
(CMI=0.060 ), i.e., they can be decoded in parallel. As gold
7

Self-Augmenting Retrieval for Diffusion Language Models
Table 3.Conditional mutual information (CMI) between adjacent
tokens in DREAM-7B reasoning traces (2WikiMultiHopQA), as
gold documents are progressively removed from the context. We
separately report entity and non-entity pairs.
All gold Gold−1Gold−2No gold
CMI (entity) 0.060 0.219 0.396 0.588
CMI (non-entity) 0.136 0.214 0.247 0.264
documents are removed, entity-pair CMI rises nearly 10× to
0.588 , while non-entity pairs increase from 0.136 to0.264 .
Grounding thus removes the inter-token dependence most
harmful to parallel decoding: tightly coupled entity spans.
This confirms that the RAG setting is well suited to parallel
decoding.
5.6. Ablations
Retriever choice.SARDIis retriever-agnostic. Replacing
BM25 with the E5-base-v2 dense retriever, SARDIstill out-
performs the strongest training-free AR baseline (ARW/
RET@1) and stays competitive with the RL-trained Search-
R1 (Table 4). Also see more detailed recall results in Ap-
pendix D. Thus, the observed behavior does not depend on
the lexical retriever.
Table 4.SARDIand strongest baselines with a dense retriever
(E5-base-v2). EM (×100),K=7.
Method 2Wiki Hotpot CofCA MuSiQue SynthW.
Search-R1 41 49 44 26 21
ARW/RET@1 47 47 41 20 22
DLMW/ SARDI(Ours) 50 52 45 22 23
Refresh schedule.SARDIrefreshes retrieval at every de-
noising step, which is wasteful when retrieval is expensive
(rerankers, large corpora, tight serving budgets). Table 5
varies the refresh frequency: refreshing every 2steps costs
only 1–2EM. Moreover, we show in Appendix E that the
retrieved set is relatively stable: on average 83–90% of doc-
uments persist between consecutive steps. Both findings
suggest that the per-step retrieval cost can be substantially
amortized – a direction we discuss in the Limitations section.
Table 5.SARDIaccuracy under different retrieval refresh frequen-
cies. EM ( ×100 ); subscripts give the change from refreshing every
step.
Refresh 2Wiki HotpotQA CofCA MuSiQue
every step 58 48 45 20
every 2 56 −2 47−1 45 20
every 4 52 −6 46−2 45 18 −26. Limitations
While SARDIis training-free in principle, the tested diffu-
sion language models do not yet reliably produce reasoning-
traces through prompting alone. This is analogous to what
Trivedi et al. (2023) observed for early medium-scale au-
toregressive models, which has since been resolved as AR
models matured; we expect the same to happen for DLMs,
removing the need for the fine-tuning step. Additionally, our
method refreshes retrieval at every denoising step, which
can incur document encoding overhead under naive im-
plementations. Because a large portion of the retrieved
documents persists between consecutive steps, integrating
Fast-dLLM-style block KV caching is a natural direction for
future work. Extending self-augmenting retrieval to latent
diffusion language models is another promising direction.
7. Conclusion
We introducedSelf-Augmenting Retrieval (SARDI), a
training-free, dynamic retrieval framework that exploits two
structural opportunities that diffusion language models open
up for retrieval-augmented generation. First, the denois-
ing trajectory exposes tentative predictions for the entire
response at every step, surfacing salient entities early and
turning them into a lookahead signal for retrieval. Second,
we have shown that grounding generation in retrieved evi-
dence sharply reduces inter-token dependence, making RAG
a regime especially well suited to parallel decoding. SARDI
leverages both: it interleaves retrieval with denoising, refin-
ing the query as the response takes shape. On five multi-hop
QA benchmarks, SARDIsubstantially improves over static
retrieval, matches or beats every training-free AR baseline,
and runs up to 8×faster. More broadly, denoising trajec-
tories offer a natural signal for dynamic retrieval, and we
expect SARDIto strengthen as diffusion language models
mature.
Acknowledgements
We thank Alexander Kreibich and Lucas He for productive
discussions that helped motivate this work. JL is supported
by a Google PhD Fellowship, LZ by a LinkedIn PhD Fel-
lowship, and DG by an Empire AI Postdoctoral Fellowship.
We are grateful to NVIDIA for access to the DGX Sta-
tion compute platform through their Early Access Program.
This work was further supported by the National Science
Foundation (NSF) under grants OAC-2118310 and 2530143,
through the AI Research Institutes program (Award No.
DMR-2433348), by funding from NewYork-Presbyterian
for the NYP-Cornell Cardiovascular AI Collaboration, and
by arXiv. Finally, we thank the anonymous reviewers,
whose feedback strengthened the paper by motivating the
analyses of early-generation recall and mutual information.
8

Self-Augmenting Retrieval for Diffusion Language Models
Impact Statement
This work studies retrieval-augmented generation for diffu-
sion language models. Improving evidence grounding can
reduce hallucinations in knowledge-intensive settings, but
the same retrieval mechanisms could produce more convinc-
ing misinformation if paired with untrusted corpora. We rec-
ommend deploying SARDIwith curated sources and prove-
nance tracking, such as citing retrieved passages alongside
generated answers. All experiments use publicly available
QA benchmarks and open retrieval corpora.
References
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H.
Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. InThe Twelfth International
Conference on Learning Representations, ICLR 2024, Vi-
enna, Austria, May 7-11, 2024. OpenReview.net, 2024.
Austin, J., Johnson, D. D., Ho, J., Tarlow, D., and van den
Berg, R. Structured Denoising Diffusion Models in Dis-
crete State-Spaces. In Ranzato, M., Beygelzimer, A.,
Dauphin, Y ., Liang, P., and Vaughan, J. W. (eds.),Ad-
vances in Neural Information Processing Systems, vol-
ume 34, pp. 17981–17993, 2021.
Fang, Y ., Yu, H.-T., Joho, H., and Fujita, S. Retrieval-
Augmented Diffusion Language Model for Generative
Commonsense Reasoning. In Jung, H., Wang, T., Toyoda,
M., Kwon, H.-Y ., and Lee, J.-w. (eds.),Database Sys-
tems for Advanced Applications, pp. 341–356, Singapore,
2026. Springer Nature Singapore. ISBN 978-981-92-
0369-7.
Gu, K., Bhat, A., Merrill, M. A., West, R., Liu, X., McDuff,
D., and Althoff, T. SynthWorlds: Controlled Parallel
Worlds for Disentangling Reasoning and Knowledge in
Language Models.arXiv preprint arXiv:2510.24427,
2025.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval Augmented Language Model Pre-Training. In
Proceedings of the 37th International Conference on Ma-
chine Learning, ICML 2020, 13-18 July 2020, Virtual
Event, Proceedings of Machine Learning Research, pp.
3929–3938. PMLR, 2020.
Ho, X., Nguyen, A. D., Sugawara, S., and Aizawa, A.
Constructing A multi-hop QA dataset for comprehen-
sive evaluation of reasoning steps. In Scott, D., Bel, N.,
and Zong, C. (eds.),Proceedings of the 28th Interna-
tional Conference on Computational Linguistics, COL-
ING 2020, Barcelona, Spain (Online), December 8-13,
2020, pp. 6609–6625. International Committee on Com-
putational Linguistics, 2020. doi: 10.18653/V1/2020.
COLING-MAIN.580.Izacard, G. and Grave, E. Leveraging Passage Retrieval
with Generative Models for Open Domain Question An-
swering. InProceedings of the 16th conference of the
european chapter of the association for computational
linguistics: main volume, pp. 874–880, 2021.
Jeong, S., Baek, J., Cho, S., Hwang, S. J., and Park, J.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented
Large Language Models through Question Complexity.
InProceedings of the 2024 Conference of the North Amer-
ican Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (NAACL), 2024.
Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-
Yu, J., Yang, Y ., Callan, J., and Neubig, G. Active re-
trieval augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing, pp. 7969–7992, 2023.
Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S. O., Wang, D.,
Zamani, H., and Han, J. Search-R1: Training LLMs to
Reason and Leverage Search Engines with Reinforcement
Learning. InSecond Conference on Language Modeling,
2025.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P. S., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense Passage
Retrieval for Open-Domain Question Answering. In
EMNLP (1), pp. 6769–6781, 2020.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks.Advances in Neural Information
Processing Systems, 33:9459–9474, 2020.
Li, X., Dong, G., Jin, J., Zhang, Y ., Zhou, Y ., Zhu, Y .,
Zhang, P., and Dou, Z. Search-o1: Agentic Search-
Enhanced Large Reasoning Models. In Christodoulopou-
los, C., Chakraborty, T., Rose, C., and Peng, V . (eds.),
Proceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, EMNLP 2025,
Suzhou, China, November 4-9, 2025, pp. 5420–5438.
Association for Computational Linguistics, 2025. doi:
10.18653/V1/2025.EMNLP-MAIN.276.
Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., and
Hashimoto, T. B. Diffusion-LM Improves Controllable
Text Generation. InAdvances in Neural Information
Processing Systems 35, 2022.
Lovelace, J., Kishore, V ., Wan, C., Shekhtman, E., and
Weinberger, K. Q. Latent Diffusion for Language Gen-
eration.Conference on Neural Information Processing
Systems (NeurIPS), 2023.
9

Self-Augmenting Retrieval for Diffusion Language Models
Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., ZHOU,
J., Lin, Y ., Wen, J.-R., and Li, C. Large Language Diffu-
sion Models. InThe Thirty-ninth Annual Conference on
Neural Information Processing Systems, 2026.
Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury,
J., Heek, J., Xiao, K., Agrawal, S., and Dean, J. Efficiently
Scaling Transformer Inference. In Song, D., Carbin, M.,
and Chen, T. (eds.),Proceedings of the Sixth Conference
on Machine Learning and Systems, MLSys 2023, Miami,
FL, USA, June 4-8, 2023. mlsys.org, 2023.
Prabhu, V . V . D. and Anand, A. DEXTER: A Benchmark for
open-domain Complex Question Answering using LLMs.
arXiv preprint arXiv:2406.17158, 2024.
Robertson, S., Zaragoza, H., et al. The Probabilistic Rele-
vance Framework: BM25 and Beyond.Foundations and
trends® in information retrieval, 3(4):333–389, 2009.
Snell, C. V ., Lee, J., Xu, K., and Kumar, A. Scaling LLM
Test-Time Compute Optimally Can be More Effective
than Scaling Parameters for Reasoning. InThe Thirteenth
International Conference on Learning Representations,
2025.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. MuSiQue: Multihop Questions via Single-hop Ques-
tion Composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving Retrieval with Chain-of-Thought Reason-
ing for Knowledge-Intensive Multi-Step Questions. In
Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.),
Proceedings of the 61st Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long
Papers), ACL 2023, Toronto, Canada, July 9-14, 2023,
pp. 10014–10037. Association for Computational Lin-
guistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.557.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D.,
Majumder, R., and Wei, F. Text Embeddings by Weakly-
Supervised Contrastive Pre-training.arXiv preprint
arXiv:2212.03533, 2022.
Wu, C., Zhang, H., Xue, S., Liu, Z., Diao, S., Zhu, L.,
Luo, P., Han, S., and Xie, E. Fast-dLLM: Training-free
acceleration of diffusion LLM by enabling KV cache
and parallel decoding. InThe Fourteenth International
Conference on Learning Representations, 2026.
Wu, J., Yang, L., Wang, Z., Okumura, M., and Zhang,
Y . CofCA: A STEP-WISE counterfactual multi-hop QA
benchmark. InThe Thirteenth International Conference
on Learning Representations, 2025.Xu, Z., Liu, Z., Yan, Y ., Wang, S., Yu, S., Zeng, Z., Xiao,
C., Liu, Z., Yu, G., and Xiong, C. ThinkNote: Enhancing
knowledge integration and utilization of large language
models via constructivist cognition modeling. In Dem-
berg, V ., Inui, K., and Marquez, L. (eds.),Findings of the
Association for Computational Linguistics: EACL 2026,
pp. 211–229, Rabat, Morocco, March 2026. Association
for Computational Linguistics. ISBN 979-8-89176-386-9.
doi: 10.18653/v1/2026.findings-eacl.12.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W., Salakhut-
dinov, R., and Manning, C. D. HotpotQA: A dataset for
diverse, explainable multi-hop question answering. In
Proceedings of the 2018 conference on empirical methods
in natural language processing, pp. 2369–2380, 2018.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . ReAct: Synergizing Reasoning and
Acting in Language Models. InThe Eleventh Interna-
tional Conference on Learning Representations, 2022.
Ye, J., Xie, Z., Zheng, L., Gao, J., Wu, Z., Jiang, X., Li,
Z., and Kong, L. Dream 7b: Diffusion large language
models.arXiv preprint arXiv:2508.15487, 2025.
Yu, C., Wang, J., Li, Y ., Chang, H., Lan, G., Sun, Q., Li,
J., Li, J., and Zhang, Z. Unlocking the Potentials of
Retrieval-Augmented Generation for Diffusion Language
Models.arXiv preprint arXiv:2601.11342, 2026.
10

Self-Augmenting Retrieval for Diffusion Language Models
A. Fine-Tuning Analysis
Table 6.Output-mode breakdown for base and fine-tuned DREAM-7B and Qwen2.5-7B on 2WikiMultiHopQA (static K=7 RAG,
n=500).
DREAM-7B Qwen2.5-7B DREAM-7B (SFT) Qwen2.5-7B (SFT)
Correct (with reasoning trace)1% 32%44%44%
Correct (no reasoning trace)27% 0% 0% 0%
Wrong72% 68% 56% 56%
To motivate the fine-tuning stage and isolate its effect from the retrieval mechanism, we present an output analysis of base
and fine-tuned models on 500examples from 2WikiMultiHopQA under static K=7 retrieval. Table 6 reports the proportion
of outputs in three categories: correct with a structured reasoning trace, correct without a reasoning trace, and wrong (which
folds in format errors such as missing the###answer marker).
Two observations are worth highlighting:
1.Base DREAM-7B cannot produce structured reasoning from prompting alone.Only 1%of base DREAM-7B
outputs include any reasoning trace at all—most simply emit a guess. The same-sized AR Qwen2.5-7B reasons in 32%
of cases. SARDIrelies on intermediate states surfacing salient entities, so this failure mode makes SARDIinapplicable
without fine-tuning.
2.Fine-tuning equalizes capability.On this n=500 subset both models land at 44% correct-with-reasoning; on the full
test set they sit within 1 EM of each other ( DLMW/RET@STATIC=43.7% ,ARW/RET@STATIC=44.5% ; Table 1).
Any subsequent gap between SARDIand matched-fine-tuning AR baselines is therefore attributable to the retrieval
mechanism rather than the underlying model.
This is analogous to what Trivedi et al. (2023) observed for early small AR models: “IRCoT relies on the base LM to have a
zero or few-shot CoT-generation ability. While this is commonly available in large LMs (over 100B), it’s not as common for
small LMs (under 20B). . . smaller LMs will likely increasingly acquire such ability.” That prediction proved correct for AR
models; we expect the same for DLMs.
A.1. Fine-Tuning Data
We construct supervised fine-tuning data from the training splits of 2WikiMultiHopQA (Ho et al., 2020) and HotpotQA (Yang
et al., 2018), augmented with synthetically generated chain-of-thought reasoning traces.
Synthetic reasoning traces.For each training example, we promptgpt-4o-miniwith the question q, gold answer a, and
supporting documents to generate a step-by-step reasoning trace that (i) identifies intermediate bridge entities, (ii) references
specific facts from the supporting documents, and (iii) concludes with the final answer in the format### [answer].
Example.The following illustrates a training example from 2WikiMultiHopQA:
Question: Where was the director of film Ronnie Rocket born?
Reasoning (synthetically generated):
Step 1: Ronnie Rocket is directed by David Lynch.
Step 2: David Lynch was born in Missoula, Montana.
### Montana
Training context.During training, we present each training example together with its gold documents, reinforcing
Dream-7B to generate reasoning traces when solving RAG problems.
11

Self-Augmenting Retrieval for Diffusion Language Models
A.2. Fine-Tuning Configuration
We fine-tune DREAM-7B using fully sharded data parallel (FSDP) on 2 ×NVIDIA B200 GPUs. Table 7 summarizes
the hyperparameters. To ensure a fair comparison, we apply the same configuration and training data to all autoregressive
baselines (Qwen2.5-7B), isolating differences attributable to the generation paradigm.
Table 7.Fine-tuning hyperparameters.
Hyperparameter Value
Base model DREAM-7B
Training epochs 3
Learning rate2×10−6
Global batch size 256
Micro batch size per GPU 16
Maximum sequence length 2048 tokens
Optimizer AdamW
Hardware 2×NVIDIA B200 GPUs
B. Prompt Templates
B.1. RAG Input Format
All models receive input in the following format:
Use ONLY the provided facts to answer the question.
Think step-by-step, then provide the final answer after the "###" marker.
Question:
{question}
Facts:
{facts}
where{question}is the input question and{facts}contains the concatenated retrieved passages.
B.2. Expected Output Format
Models are trained to produce outputs in the following format:
Step 1: [First reasoning step grounded in documents]
Step 2: [Second reasoning step]
...
### [final answer]
This format serves two purposes: it enables extraction of intermediate entities from partial reasoning traces for self-
augmenting retrieval, and it provides a consistent extraction point (###) for evaluation.
C. Additional Experimental Details
Evaluation protocol.We extract the final answer by parsing text following the ### marker. Exact Match (EM) is
computed after normalizing both predicted and gold answers and checking for string equality.
Retrieval configuration.We use BM25 (Robertson et al., 2009) for all experiments unless otherwise specified. For the
dense-retriever results in Section 5.6 we use E5-base-v2 (Wang et al., 2022). For SARDI, retrieval is performed at every
denoising iteration using the concatenation of the original question and the current intermediate response as the query, with
K=7passages per iteration unless otherwise specified.
12

Self-Augmenting Retrieval for Diffusion Language Models
Search-R1 setup.The four AR baselines (ARW/RET@STATIC, ARW/RET@10, ARW/RET@1, ARW/
RET@ADAPTIVE) and the training-free agentic baselines (AdaptiveRAG, ReAct) all share the same Qwen2.5-7B backbone,
supervised-fine-tuned on the combined training sets of 2WikiMultiHopQA and HotpotQA (Appendix A); any accuracy gap
with SARDItherefore isolates the retrieval mechanism. Search-R1 is the only baseline that doesnotshare this backbone: we
evaluate the authors’ publicly released checkpoint, which they trained via PPO on HotpotQA using a multi-stage reward (Jin
et al., 2025). Reproducing this RL training pipeline was not feasible within our compute budget, so we accept this additional
confound and treat Search-R1 as an RL-trained reference point rather than a controlled comparison (hence the gray styling
in Table 1).
D. Per-Method Recall: Full Tables
Figure 5 in the main text reports gold-document recall over generation progress for SARDI and the AR retrieval baselines
that fire at every (or every- Nth) token. For completeness, this section reports the full per-method numbers, including the
agentic methods (AdaptiveRAG, ReAct, Search-R1) and AR speculative-lookahead (FLARE).
We report two complementary recall metrics:
•Per-query final recall(Table 8): fraction of gold documents present in the retrieved set Dat thefinalretrieval step,
averaged over questions. This is the metric most directly tied to answer accuracy on a single forward pass.
•Cumulative recall(Table 9): fraction of gold documents that have appeared in Datanypoint during generation, i.e., the
union over all retrieval steps. This view is fairer to methods that issue few but targeted queries.
Caveat: cumulative recall depends on retrieval volume.A method that issues many queries (e.g., SARDI, AR ret@1)
trivially accumulates more unique documents than one that issues few targeted queries (e.g., Search-R1), and cumulative
recall therefore isnotdirectly comparable across methods without controlling for total documents touched. Table 9
accordingly reports the average number of unique documents retrieved per question as a subscript next to each cumulative-
recall value, so the reader can normalize.
Table 8.Per-query gold-document recall at the final retrieval step ( ×100 ). Methods that issue specific targeted queries (AdaptiveRAG,
ReAct, AR ret@adaptive/FLARE, Search-R1) retrieve few documents per query, so per-query recall is low by design even when their
cumulative coverage (Table 9) is comparable. SARDImatches AR ret@1 acrossKand retrievers.
2Wiki MuSiQue
BM25 Dense BM25 Dense
MethodK=3K=7K=10K=15K=3K=7K=10K=15K=3K=7K=10K=15K=3K=7K=10K=15
AdaptiveRAG 5 5 4 3 – 5 4 3 9 10 9 7 – 9 – –
ReAct 38 48 51 54 27 31 31 33 26 34 37 40 27 31 32 35
AR ret@adaptive (FLARE) 22 27 29 32 14 17 18 19 14 20 22 25 14 20 21 22
AR ret@10 65 78 82 86 54 61 62 63 42 53 56 61 38 47 51 54
AR ret@1 66 80 85 88 54 61 63 64 43 54 58 62 38 48 51 54
Search-R1 42 53 57 62 28 31 33 35 29 38 42 45 28 35 37 41
SARDIτ c=0.966 81 83 86 55 63 62 63 43 55 56 60 39 50 53 55
SARDIτ c=0.9567 81 84 88 56 63 64 65 44 56 59 64 40 50 53 56
E. Document Persistence Across Refresh Steps
Table 5 relies on the observation that retrieved documents change only gradually between consecutive denoising steps,
so document-level KV-cache reuse is feasible. We measure the average overlap between consecutive retrieved sets as
|Dt∩Dt−1|/|Dt−1|:
Dataset % docs retained per step
2WikiMultiHopQA88%
HotpotQA83%
CofCA89%
MuSiQue84%
SynthWorlds-RM90%
13

Self-Augmenting Retrieval for Diffusion Language Models
Table 9.Cumulative gold-document recall ( ×100 , main number) and average number of unique documents retrieved per question ( subscript ).
Cumulative recall measures the union of all gold documents surfaced at any point during generation, but isnot directly comparable across
methods that retrieve different total document counts: a method that issues many queries (e.g., SARDI, AR ret@1) trivially accumulates
more docs than one that issues few (e.g., Search-R1). The subscripts let the reader normalize. Even controlling for total docs, SARDI
remains competitive with the strongest baselines.
2Wiki MuSiQue
BM25 Dense BM25 Dense
MethodK=3K=7K=10K=15K=3K=7K=10K=15K=3K=7K=10K=15K=3K=7K=10K=15
AdaptiveRAG 71 6 7913 7714 6215 – – – – 48 7 5613 5514 4815 – – – –
ReAct 73 7 8115 8221 8331 577 5715 5721 5831 506 6015 6322 6533 547 6016 6222 6433
AR ret@adaptive (FLARE) 73 8 8220 8429 8743 639 6622 6732 6849 519 6222 6531 6948 529 6022 6432 6548
AR ret@10 71 6 8315 8622 8934 616 6615 6722 6833 496 6016 6423 6834 466 5515 5822 6232
AR ret@1 73 7 8520 8929 9146 649 6823 6933 7050 528 6321 6730 7145 509 5922 6231 6546
Search-R1 74 6 8516 8823 9034 607 6317 6325 6337 577 6516 6823 – 59 7 6517 6723 7034
SARDIτ c=0.970 5 8314 8520 8831 616 6817 6624 6737 477 6319 6128 6544 477 5819 5928 6142
SARDIτ c=0.9571 5 8314 – 90 33 637 6818 7026 7140 527 6420 6830 7346 498 5920 6229 6644
Since 83–90% of documents persist between consecutive steps, only a small fraction needs recomputation, making
document-level KV caching similar to Fast-dLLM (Wu et al., 2026) an attractive direction for combining SARDIwith
diffusion-decoding-acceleration techniques.
F. Accuracy vs. Throughput: Full Datasets
Figure 3 in the main text shows the Pareto frontier on 2WikiMultiHopQA and CofCA. Figure 6 reports the same plot
across all four benchmarks. SARDIdominates the autoregressive iterative-retrieval baselines on every dataset and matches
Search-R1 at3–8×lower latency.
0 1 2 3 4 5 6 7
Questions per second 
35404550556065Exact match (%) 
0.60.70.80.90.95
ret@staticret@1
ret@10
ret@adaptive2WikiMultihopQA
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
0 1 2 3 4 5
Questions per second 
35.037.540.042.545.047.550.052.5Exact match (%) 
0.60.70.80.90.95
ret@staticret@1
ret@10
ret@adaptiveHotpotQA
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
Questions per second 
121416182022242628Exact match (%) 
0.60.70.80.90.95
ret@staticret@1
ret@10ret@adaptiveMuSiQue
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
0 1 2 3 4 5
Questions per second 
30.032.535.037.540.042.545.047.5Exact match (%) 
0.60.70.80.90.95
ret@static
ret@1
ret@10
ret@adaptiveCofCA
SARDI (c [0.60,0.95])
AR baselines
ReAct
Search-R1 (RL trained)
Figure 6.Accuracy vs. throughput trade-off across all four benchmarks, companion to Figure 3.
14

Self-Augmenting Retrieval for Diffusion Language Models
Table 10.Threshold-based unmasking matches fixed-step accuracy at2–3×the speed. EM (×100) and time per example (s).
2Wiki HotpotQA CofCA MuSiQue
EM / T EM / T EM / T EM / T
50 fixed steps 59 / 1.33 47 / 1.46 43 / 1.48 20 / 1.42
τc=0.958 −1/ 0.39−0.94 48+1/ 0.64−0.82 45+2/ 0.75−0.73 20 / 0.88 −0.54
τc=0.9559 / 0.56 −0.77 49+2/ 0.90−0.56 45+2/ 1.09−0.39 21+1/ 1.19−0.23
Table 11.Query-threshold sweep, full numbers. EM (%) on the four datasets withτ qsweeps, at commit thresholdsτ c∈ {0.9,0.95}.
Datasetτ cτq=0τ q=0.1τ q=0.2τ q=0.3τ q=0.4τ q=0.5τ q=0.6τ q=0.7τ q=0.8τ q=0.9
2WikiMultihopQA0.9 57.8 58.1 57.4 57.3 56.6 55.6 54.7 54.5 54.0 53.2
0.95 59.1 59.4 58.7 58.4 57.6 57.1 56.2 55.6 54.8 54.0
HotpotQA0.9 48.3 48.1 47.6 46.4 45.9 46.0 44.8 44.7 44.4 44.1
0.95 48.7 48.4 47.8 47.0 46.3 45.6 45.3 44.9 44.4 44.1
MuSiQue0.9 20.5 20.6 19.7 19.2 18.2 16.7 16.3 15.7 15.7 15.2
0.95 20.6 20.6 19.5 18.9 17.9 17.1 16.3 16.3 15.6 14.7
SynthWorlds0.9 21.1 21.4 21.1 19.9 17.9 17.5 17.9 17.0 16.8 17.2
0.95 21.7 21.5 21.3 19.8 19.5 18.2 17.5 17.5 17.0 16.9
G. Threshold-Based vs. Fixed-Step Decoding
Table 10 compares fixed-step decoding ( T=50 ) against threshold-based unmasking at the two commit thresholds used in the
main paper,τ c=0.9andτ c=0.95. Threshold decoding matches fixed-step accuracy at up to3×the speed.
H. Query-Threshold Sweep: Full Numbers
Figure 4 in the main text plots the τqsweep on 2WikiMultiHopQA. Table 11 reports full EM numbers across all four
benchmarks for which τqsweeps were run, at both commit thresholds τc∈ {0.9,0.95} and ten coarse τqvalues spanning
the spectrum from aggressive lookahead ( τq=0, the default SARDIsetting) to the conservative endpoint ( τq=0.9 , querying
only on near-committed tokens). The pattern observed on 2WikiMultiHopQA reproduces on HotpotQA, MuSiQue, and
SynthWorlds-RM: peak accuracy lands at τq∈[0,0.1] on every (dataset, τc) cell, and EM generally declines as the query
threshold tightens. The conservative endpoint loses 4–6EM relative to the default, consistent with our hypothesis that
retrieval benefits from speculative future tokens that are not yet stable enough to commit.
I. Qualitative Example: Threshold-Based Unmasking
A central hypothesis of this work is that RAG exhibits a structure uniquely amenable to parallel decoding: when retrieved
evidence is sufficiently informative, many output tokens become conditionally independent given the evidence and can be
committed simultaneously. Threshold-based unmasking (Equation (6)) exploits this by committing all high-confidence
tokens at once rather than artificially limiting parallelism. The following example illustrates this behavior for τc=0.8 ,
requiring only 2 steps:
Where was the place of death of the director of Fight Of The Tertia?
x2=“[M][M][M][M][M][M][M][M][M][M][M][M][M]”
D2=“Fight of the Tertia is a 1952 West German family film directed by Erik Ode [. . . ]”
x1=“(1) Fight of the Tertia is directed by Erik Ode. (2) Erik Ode died in [M][M][M]”
D1=“Erik Ode (born Fritz Erik Signy Odemar, 6 November 1910 in Berlin, died 19 July 1983 in Kreuth-Weißach) was a [. . . ]”
x0 =“(1) Fight of the Tertia is directed by Erik Ode. (2) Erik Ode died in Kreuth. ###
Kreuth”
15