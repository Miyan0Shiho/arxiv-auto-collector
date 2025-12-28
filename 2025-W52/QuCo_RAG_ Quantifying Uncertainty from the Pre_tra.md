# QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation

**Authors**: Dehai Min, Kailin Zhang, Tongtong Wu, Lu Cheng

**Published**: 2025-12-22 08:28:05

**PDF URL**: [https://arxiv.org/pdf/2512.19134v1](https://arxiv.org/pdf/2512.19134v1)

## Abstract
Dynamic Retrieval-Augmented Generation adaptively determines when to retrieve during generation to mitigate hallucinations in large language models (LLMs). However, existing methods rely on model-internal signals (e.g., logits, entropy), which are fundamentally unreliable because LLMs are typically ill-calibrated and often exhibit high confidence in erroneous outputs. We propose QuCo-RAG, which shifts from subjective confidence to objective statistics computed from pre-training data. Our method quantifies uncertainty through two stages: (1) before generation, we identify low-frequency entities indicating long-tail knowledge gaps; (2) during generation, we verify entity co-occurrence in the pre-training corpus, where zero co-occurrence often signals hallucination risk. Both stages leverage Infini-gram for millisecond-latency queries over 4 trillion tokens, triggering retrieval when uncertainty is high. Experiments on multi-hop QA benchmarks show QuCo-RAG achieves EM gains of 5--12 points over state-of-the-art baselines with OLMo-2 models, and transfers effectively to models with undisclosed pre-training data (Llama, Qwen, GPT), improving EM by up to 14 points. Domain generalization on biomedical QA further validates the robustness of our paradigm. These results establish corpus-grounded verification as a principled, practically model-agnostic paradigm for dynamic RAG. Our code is publicly available at https://github.com/ZhishanQ/QuCo-RAG.

## Full Text


<!-- PDF content starts -->

QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for
Dynamic Retrieval-Augmented Generation
Dehai Min1, Kailin Zhang2, Tongtong Wu3, Lu Cheng1
1University of Illinois at Chicago,2New York University,3Monash University
dmin10@uic.edu, kz2739@nyu.edu, tongtong.wu@monash.edu, lucheng@uic.edu
Abstract
Dynamic Retrieval-Augmented Generation
adaptively determines when to retrieve dur-
ing generation to mitigate hallucinations in
large language models (LLMs). However,
existing methods rely on model-internal sig-
nals (e.g., logits, entropy), which are funda-
mentally unreliable because LLMs are typi-
cally ill-calibrated and often exhibit high con-
fidence in erroneous outputs. We propose
QuCo-RAG, which shifts fromsubjectivecon-
fidence toobjectivestatistics computed from
pre-training data. Our method quantifies uncer-
tainty through two stages: (1) before genera-
tion, we identify low-frequency entities indicat-
ing long-tail knowledge gaps; (2) during gen-
eration, we verify entity co-occurrence in the
pre-training corpus, where zero co-occurrence
often signals hallucination risk. Both stages
leverage Infini-gram for millisecond-latency
queries over 4 trillion tokens, triggering re-
trieval when uncertainty is high. Experiments
on multi-hop QA benchmarks show QuCo-
RAG achieves EM gains of 5–12 points over
state-of-the-art baselines with OLMo-2 models,
and transfers effectively to models with undis-
closed pre-training data (Llama, Qwen, GPT),
improving EM by up to 14 points. Domain gen-
eralization on biomedical QA further validates
the robustness of our paradigm. These results
establish corpus-grounded verification as a prin-
cipled, practically model-agnostic paradigm for
dynamic RAG1.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Gao et al., 2023b) mitigates LLM
hallucinations by grounding generation in exter-
nal evidence. Early RAG systems employ static
strategies with a single retrieval step before genera-
tion (Karpukhin et al., 2020; Shi et al., 2024; Min
et al., 2025), but fall short for complex multi-step
1Our code is publicly available at https://github.com/
ZhishanQ/QuCo-RAG.
Il    Seduttore   was   directed   by   Mario   Camerini  and…
Pre-training CorpusEngineEntity Co-occurrence frequency: 0millisecondsPotential HallucinationOutputs:(b) QuCo-RAG: Quantifying Uncertainty via Corpus Statistics
Question: Are the directors of Il Seduttore and The Trial of Joan of Arc from the same country?Il      Seduttore   was   directed   by   Mario   Camerini   and…HighUncertaintyLowUncertainty(a) DRAGIN: Token-level Uncertainty via Attention & EntropyOutputs:𝑼𝒏𝒄𝒆𝒓𝒕𝒂𝒊𝒏𝒕𝒚=	𝟏.𝟒𝟕	>	𝒕𝒉𝒓𝒆𝒔𝒉𝒐𝒍𝒅𝑼𝒏𝒄𝒆𝒓𝒕𝒂𝒊𝒏𝒕𝒚<	𝒕𝒉𝒓𝒆𝒔𝒉𝒐𝒍𝒅
❌
❌( Correct: Franco Rossi )( Token from the question )
✅
Query
Figure 1: Comparison of retrieval triggering mecha-
nisms. (a) DRAGIN relies on model-internal signals,
incorrectly assigning high uncertainty to “Il” (a token
from the question) while showing low uncertainty on the
hallucinated director name. (b) QuCo-RAG correctly de-
tects the hallucination through zero entity co-occurrence
in the pre-training corpus.
tasks where information needs emerge dynamically
during generation (Su et al., 2025; Wang et al.,
2025, 2023). This has driven the emergence of
Dynamic RAG methods that adaptively determine
when and what to retrieve based on the generation
process (Jiang et al., 2023; Asai et al., 2024).
Current dynamic RAG methods predominantly
rely on quantifying uncertainty through model-
internal signals such as token probability (Jiang
et al., 2023) or entropy (Su et al., 2024; Li et al.,
2025a). However, these methods assume internal
signals reliably indicate generation correctness—
an assumption that is fundamentally flawed (Li
et al., 2024b). As illustrated in Figure 1(a), the no-
table work DRAGIN (Su et al., 2024) exhibits low
uncertainty when generating the incorrect director
name “Mario Camerini”, yet assigns high uncer-
tainty to “Il”—a token from the question. This
failure reflects a well-documented problem: LLMs
are poorly calibrated (Guo et al., 2017; Kadavath
1arXiv:2512.19134v1  [cs.CL]  22 Dec 2025

et al., 2022; Achiam et al., 2023)—their confidence
scores fail to correlate with actual prediction accu-
racy. This miscalibration leads to “confident hallu-
cinations,” where models produce incorrect content
with high confidence (Tian et al., 2023). Further-
more, post-training techniques such as SFT (Dong
et al., 2024) and Reinforcement Learning (Ouyang
et al., 2022; Guo et al., 2025) often exacerbate
this by encouraging decisive answers. More funda-
mentally, recent theoretical work (Kalai and Vem-
pala, 2024) further shows that for rarely-seen facts,
even perfectly calibrated models must hallucinate
to maintain statistical consistency.
To bypass the limitations, we proposeQuCo-
RAG, a framework that determines when to re-
trieve byQuantifying uncertainty via pre-training
Corpus statistics, shifting from subjective internal
confidence to objective external evidence. Our key
insight is that an LLM’s factual knowledge is funda-
mentally shaped by its pre-training corpus (Balepur
et al., 2025): low-frequency entities correspond to
long-tail knowledge that models struggle to mem-
orize reliably, while zero co-occurrence between
entity pairs indicates the model has no evidential
basis for claims relating them. Based on this in-
sight, QuCo-RAG operates through two-stage de-
tection:(1) Pre-Generation Knowledge Assess-
ment:We query entity frequencies in the pre-
training corpus, triggering retrieval when entities
are low-frequency (long-tail knowledge risks).(2)
Runtime Claim Verification:We extract knowl-
edge triplets from each generated sentence and ver-
ify entity co-occurrence; zero co-occurrence trig-
gers retrieval and regeneration. Both stages lever-
age Infini-gram (Liu et al., 2024) for millisecond-
latency queries over trillion-token corpora.
To validate our approach, we first evaluate QuCo-
RAG on multi-hop QA benchmarks using the
OLMo-2 model family (7B, 13B, 32B) (OLMo
et al., 2024), which provides full access to its 4-
trillion token pre-training corpus for precise statisti-
cal verification. Results show QuCo-RAG achieves
5–12 point improvements on Exact Match (EM)
over state-of-the-art baselines across all model
scales, while maintaining competitive efficiency.
Beyond this matched-corpus setting, we demon-
strate QuCo-RAG’s broad applicability through
two additional dimensions of evaluation. First,
forcross-model transferability, we show that
corpus statistics computed from OLMo-2’s pre-
training corpus serve as effective proxies for mod-
els with undisclosed training data. Leveraging thesubstantial overlap of web-scale pre-training cor-
pora, QuCo-RAG yields up to 14 EM improve-
ments on Llama-3, Qwen2.5, and GPT-4.1/5 se-
ries. Second, fordomain generalization, we eval-
uate on PubMedQA (Jin et al., 2019), a biomed-
ical QA benchmark requiring specialized knowl-
edge. QuCo-RAG achieves the best accuracy while
internal-signal methods either trigger excessive re-
trievals or fail to improve over no-retrieval base-
lines, demonstrating that our framework general-
izes robustly without domain-specific tuning.
2 Related Work
Dynamic Retrieval-Augmented LLMDynamic
RAG methods have evolved to address the limita-
tions of static retrieval approaches by adaptively
determining when and what to retrieve during gen-
eration (Xu et al., 2024; Yu et al., 2024; Yang et al.,
2025). FLARE (Jiang et al., 2023) pioneered this
direction by triggering retrieval when encounter-
ing low-probability tokens. Self-RAG (Asai et al.,
2024) extended this paradigm by training models
to generate special reflection tokens that assess
retrieval necessity and response quality, though
requiring additional fine-tuning. More recent ap-
proaches (Ma et al., 2025) construct more sophis-
ticated uncertainty metrics: DRAGIN (Su et al.,
2024) integrates multiple model-internal signals
including entropy and attention weights, ETC (Li
et al., 2025a) considers first- and second-order en-
tropy differences to capture uncertainty trends, and
SeaKR (Yao et al., 2025) extracts self-aware uncer-
tainty from LLMs’ internal FFN states. However,
these methods all rely on model-internal signals,
which may not reliably indicate correctness.
Reusing LLM Pre-Training Data at Inference
TimeRecent work explores unlocking additional
value from pre-training corpora at inference time.
Fang et al. (2025) showed that retrieving from
the model’s own pre-training data yields perfor-
mance gains equivalent to a 5×increase in pre-
training compute. Efficient infrastructure has
emerged to support trillion-scale corpus access.
Infini-gram (Liu et al., 2024) provides millisecond-
latency n-gram counting via suffix arrays, while
Infini-gram mini (Xu et al., 2025) reduces index
size to 44% of the corpus via FM-index (Ferragina
and Manzini, 2000). OLMoTrace (Liu et al., 2025)
enables real-time tracing of LLM output back to
verbatim matches in training documents. Our work
leverages this infrastructure for a distinct purpose:
2

Stage 1. Pre-Generation Knowledge Assessment Entity Extraction‘Silas Hardy’ ,‘Lee Mantle’Silas Hardy : 258Lee Mantle : 180Frequency
Pre-training CorpusIndex Engine𝑭𝒓𝒆𝒒𝒂𝒗𝒈<𝝉𝒆𝒏𝒕𝒊𝒕𝒚Retrieved docs[1]: Silas Hardywas an..[2]: ...…
𝑆): Lee Mantle was born on December 13, 1851.𝑆*: Silas Hardy was born on January 24, 1827.CheckCo-occurrenceCount: 3Count: 0
LLM Generation Loop:
Who was born earlier, Silas Hardy or Lee Mantle?Question: 
Stage 2. Runtime Claim Verification
<𝝉𝒄𝒐𝒐𝒄𝑆*∗: Silas Hardy was born on 30 April 1867.Retriever
Regeneration with retrieved docs…(corrected)Query: "Silas Hardy" + "born on"
Triplet Extractor( Lightweight )if triggered(4T Tokens)
High Output UncertaintyRetrieverHigh Input Uncertainty
LLM4T tokens
Figure 2: Overview of QuCo-RAG Framework.
using pre-training corpus statistics toquantify un-
certainty and trigger retrieval, enabling reliable
hallucination detection and mitigation.
3 Methodology
3.1 Problem Formulation
We formalize the dynamic RAG problem as follows.
LetMdenote an LLM, Crepresent an external
knowledge base for retrieval (e.g., Wikipedia), and
Pdenote the pre-training corpus used to train M.
Given an input question Q, the model generates
a response y= (s 1, s2, . . . , s N), where siis the
i-th generated sentence. A dynamic RAG system
makes two critical decisions during generation:
(1) When to retrieve.At each step i, determine
whether to trigger retrieval:
δi=f trigger(Q, s <i; Θ)∈ {0,1},(1)
whereΘdenotes the source of uncertainty signals.
Unlike prior methods that rely on internal model
states (i.e., Θ =M ), we ground the decision in
pre-training corpus statistics (i.e.,Θ =P).
(2) What to retrieve.When δi= 1, construct
a query qi=f query(Q, s <i)and retrieve related
documents Di=Retrieve(q i,C), where fquery is
the query formulation function.
Binary Nature of Retrieval Decisions.Note that
the retrieval decision δi∈ {0,1} is inherently bi-
nary: the system either retrieves or not. This obser-
vation motivates our design: rather than estimating
continuous confidence scoresfrom model-internal
signals to infer uncertainty, whose thresholds lack
clear semantic grounding, we directly leveragedis-
crete corpus statisticsto determine whether themodel faces high uncertainty (retrieve) or low un-
certainty (proceed without retrieval). Specifically,
we consider two high-uncertainty scenarios:(1)
Input uncertainty: the question contains entities
rarely seen during pre-training, indicating insuf-
ficient knowledge coverage;(2) Output uncer-
tainty: the generated claim relates entities that
never co-occur in the corpus, indicating lack of
evidential support. Both signals are grounded in
corpus statistics, as illustrated in Figure 2.
3.2 Pre-Generation Knowledge Assessment
To quantify input uncertainty, we employ a pre-
check mechanism before generation begins. We
first use a lightweight entity extractor to identify
a set of key entities EQ={e 1, e2, . . . , e m}from
the input question Q. For each entity e∈ E Q, we
query its frequency in the pre-training corpus P,
denoted as freq(e;P) . We posit that entities with
low frequency in Prepresent long-tail knowledge
risks, where the model is likely to hallucinate. Re-
trieval is triggered if the average entity frequency
falls below a predefined threshold:
δpre=I 
Avge∈EQfreq(e;P)< τ entity
.(2)
We set τentity= 103as the default threshold; results
remain stable across a wide range ( 103to107) as
shown in Appendix A.2. If δpre= 1, we use the
original question Qas the search query to retrieve
relevant documents D0, which are prepended to the
model’s context before generation starts.
3.3 Runtime Claim Verification
To quantify output uncertainty, QuCo-RAG con-
tinuously monitors each generated sentence siby
3

verifying whether the claimed facts have evidential
support in the pre-training corpus. For a generated
sentence si, we extract a set of knowledge triplets
T={(h, r, t)} , where h,r,trepresent the head
entity, relation, and tail entity, respectively. We
quantify the evidential support for each triplet by
computing the co-occurrence frequency of the head
and tail entities within a defined window ω(e.g., a
document or paragraph) inP:
cooc(h, t;P) =|{ω∈ P:h∈ω∧t∈ω}|.(3)
We compute cooc(h, t) rather than cooc(h, r, t)
because relational predicates exhibit high lexi-
cal variability (e.g., “employed by” vs. “worked
at”), while named entities are more lexically sta-
ble (Galárraga et al., 2014). Retrieval is triggered
if the co-occurrence count falls below a threshold
τcooc(default set to 1):
δi=I
min
(h,r,t)∈Tcooc(h, t;P)< τ cooc
.(4)
The rationale for τcooc= 1is intuitive: if two enti-
ties never co-occur in the pre-training corpus, the
generated claim lacks evidential support and likely
constitutes a hallucination (Mallen et al., 2023;
Kandpal et al., 2023). Notably, co-occurrence evi-
dence isasymmetric: while cooc(h, t;P)>0 does
not guarantee correctness (entities may co-occur
with different relations or in unrelated contexts),
cooc(h, t) = 0 strongly indicates hallucination
risk (Gao et al., 2023a; Ravichander et al., 2025).
When retrieval is triggered ( δi= 1), we construct
aSemantic-Oriented Queryusing the head entity
and relation ( q=h⊕r ) to retrieve supporting
documents and regenerate the sentence.
3.4 Implementation Details
Corpus Statistics via Infini-gram.We leverage
Infini-gram (Liu et al., 2024), a suffix array-based
engine that supports millisecond-latency queries
over trillion-token corpora, enabling real-time com-
putation during generation.
Lightweight Triplet Extraction.To minimize
overhead while ensuring extraction quality, we
distill a specialized 0.5B model from GPT-4o-
mini (Hurst et al., 2024). Specifically, we construct
40K annotated examples using in-context learning,
then perform full-parameter supervised fine-tuning
on Qwen2.5-0.5B-Instruct (Team, 2024). Repre-
sentative training examples are provided in Ap-
pendix A.3.4 Experimental Setup
4.1 Datasets and Implementation
We evaluate on two widely adopted knowledge-
intensive multi-hop QA benchmarks: 2WikiMul-
tihopQA (Ho et al., 2020) and HotpotQA (Yang
et al., 2018). Following Su et al. (2024), we
sample the first 1,000 validation examples from
each as our test sets and report Exact Match (EM)
and token-level F1 score as evaluation metrics,
which are well-suited for these benchmarks as
answers are short-form entities that can be re-
liably extracted and matched. Prior work (Li
et al., 2025a) has shown that EM/F1 conclusions
align with LLM-as-judge (Li et al., 2025b) evalu-
ations on these datasets. For retrieval, we employ
BM25 (Robertson et al., 2009) over the Wikipedia
dump from Karpukhin et al. (2020) as our exter-
nal corpus C, retrieving top-3 documents per query.
We also verify robustness with dense retrievers in
Appendix A.4. In our experiments, we query entity
frequencies and co-occurrences via the Infini-gram
API2, which hosts the full OLMo-2 pre-training cor-
pus index. We set the co-occurrence window size
to 1,000 tokens, roughly matching passage-level
context length. More detailed LLM generation set-
tings and the full prompt template are provided in
Appendix A.1. All experiments are conducted on
NVIDIA H200 GPUs (141GB HBM3e).
4.2 Baselines
No Retrieval: Wo-RAGgenerates answers di-
rectly without any external retrieval, serving as
the lower bound to measure RAG benefits.
Static Retrieval:Single-Round RAG (SR-RAG):
performs one-time retrieval using the input ques-
tion before generation begins. Fixed-Sentence
RAG (FS-RAG) (Trivedi et al., 2023) triggers re-
trieval after every generated sentence, using the last
sentence as the query.
Dynamic Retrieval: FLARE(Jiang et al., 2023)
triggers retrieval on low-probability tokens.DRA-
GIN(Su et al., 2024) combines entropy, atten-
tion, and semantic signals.ETC(Li et al., 2025a)
models first- and second-order entropy differences.
SeaKR(Yao et al., 2025) leverages internal FFN
states for uncertainty estimation. All baseline re-
sults are reproduced using their released code.
2API Endpoint Documentation: https://infini-gram.
readthedocs.io/en/latest/api.html . The Infini-gram in-
dex supports local deployment for offline environments, requir-
ing primarily CPU and disk storage rather than GPU resources.
4

Table 1: Performance comparison on multi-hop QA benchmarks across OLMo-2 model scales.Bold: best; underline :
second-best.Improv.: absolute gain over best baseline. 2Wiki: 2WikiMultihopQA.
OLMo-2-7B OLMo-2-13B OLMo-2-32B
2Wiki HotpotQA 2Wiki HotpotQA 2Wiki HotpotQA
Method EM F1 EM F1 EM F1 EM F1 EM F1 EM F1
Wo-RAG 20.1 26.4 22.6 31.6 28.5 34.5 24.4 33.6 33.3 40.3 22.0 31.3
SR-RAG 23.7 30.7 29.7 40.7 28.9 35.7 29.7 39.5 37.4 46.5 29.5 40.4
FS-RAG 21.1 28.3 14.5 20.7 28.8 35.1 14.6 21.9 34.6 41.0 13.9 19.5
FLARE 22.9 28.9 20.3 28.4 26.2 31.5 15.3 21.9 32.0 39.3 28.3 39.8
DRAGIN 22.8 29.0 17.5 24.7 28.5 33.9 19.5 27.6 33.3 40.2 17.7 24.3
ETC 23.4 29.8 25.1 34.7 29.7 35.9 29.3 39.5 36.0 43.6 30.8 42.2
SeaKR 25.3 32.7 24.8 35.0 29.6 34.6 26.2 37.3 30.2 38.2 28.7 40.4
QuCo-RAG 32.7 41.1 35.3 46.1 41.7 49.1 35.0 46.8 46.8 56.2 41.6 54.2
Improv. +7.4 +8.4 +5.6 +5.4 +12.0 +13.2 +5.3 +7.3 +9.4 +9.7 +10.8 +12.0
4.3 Models
Primary Models (Matched Corpus).We use
the OLMo-2-Instruct family (OLMo et al., 2024)
(7B, 13B, and 32B) as our primary evaluation
targets. OLMo-2 achieves performance compet-
itive with mainstream models like Qwen2.5 while
providing publicly available training data, code,
and recipes. The pre-training corpus3comprises
about 4 trillion tokens from diverse sources. This
transparency enables precise computation of entity
frequencies and co-occurrence statistics, making
OLMo-2 ideal for validating our method.
Transferability Models (Proxy Corpus).A key
advantage of QuCo-RAG is its applicability to
LLMs with undisclosed pre-training data. Given
that web-scale pre-training corpora share substan-
tial overlap (Soldaini et al., 2024), statistics de-
rived from a transparent and comprehensive corpus
can serve as effective proxies for other models.
We demonstrate this by using the OLMo-2 cor-
pus as a proxy for Llama-3-8B-Instruct (Grattafiori
et al., 2024), Qwen2.5-32B-Instruct (Team, 2024),
and proprietary models (GPT-4.1 (OpenAI, 2025a),
GPT-5-chat (OpenAI, 2025b)). For GPT models,
we additionally compare against their built-in agen-
tic web search, where the model autonomously
invokes web search via the Responses API.
5 Experimental Results
We design experiments to answer three core re-
search questions:
•RQ1:How does corpus-based uncertainty com-
pare to model-internal signals? (§5.1)
3https://huggingface.co/datasets/allenai/
olmo-mix-1124•RQ2:How well does QuCo-RAG transfer to
models with undisclosed training data? (§5.2)
•RQ3:What is the efficiency-performance trade-
off of QuCo-RAG? (§5.3)
5.1 Main Results (RQ1)
Table 1 presents the main results on OLMo-2 mod-
els across both benchmarks.
QuCo-RAG Achieves Significant Improvements
over Baselines.Across all model scales and
datasets, QuCo-RAG consistently outperforms the
strongest baselines by significant margins. On
OLMo-2-7B, QuCo-RAG achieves 32.7 EM on
2WikiMultihopQA and 35.3 EM on HotpotQA, sur-
passing the best baseline by +7.4 and +5.6 points
respectively. The improvements become even more
pronounced with larger models: OLMo-2-13B
shows gains of +12.0 EM on 2WikiMultihopQA,
while OLMo-2-32B achieves +10.8 EM improve-
ments on HotpotQA. These results demonstrate
that grounding retrieval decisions in corpus statis-
tics provides a fundamentally more reliable signal
than model-internal uncertainty measures.
Internal-Signal Methods Show Inconsistent Per-
formance.Methods relying on model-internal
signals (FLARE, DRAGIN, ETC, SeaKR) show
highly variable results across settings. For instance,
ETC achieves second-best performance in some
configurations, yet underperforms even simple SR-
RAG in others. DRAGIN achieves only 17.5–19.5
EM on HotpotQA across all model sizes, substan-
tially underperforming SR-RAG. This inconsis-
tency stems from the fundamental unreliability of
internal uncertainty signals. A detailed case study
is provided in Appendix A.5.
5

Wo-RAG SR-RAG FS-RAGFLAREDRAGINETC
SeaKR
QuCo-RAG0100200300400Token Consumption (Bars)(a) Token Efficiency vs Performance
Wo-RAG SR-RAG FS-RAGFLAREDRAGINETC
SeaKR
QuCo-RAG0246810LLM Calls (Bars)(b) LLM Calls vs Performance
0 2 4 6
Average Retrieval Operations20304050EM Score(c) Performance vs Retrieval Cost
Wo-RAG
SR-RAG
FS-RAG
FLARE
DRAGIN
ETC
SeaKR
QuCo-RAG
1520253035
EM Score
1520253035
EM ScoreBaselines (Bar, Left Axis) QuCo-RAG (Bar, Left Axis) Baselines EM (Right Axis) QuCo-RAG EM (Right Axis)Figure 3: Efficiency-performance trade-off analysis on HotpotQA with OLMo-2-13B-Instruct. (a) EM score versus
Token consumption. (b) EM score versus LLM calls. (c) Performance versus Retrieval frequency. QuCo-RAG
achieves the highest EM with moderate token usage and LLM calls.
Table 2: Transferability to other model families (EM
scores). HPQA: HotpotQA. ‘-’ indicates the method is
not applicable due to API limitations. Full results with
F1 score are in Appendix A.6.
Qwen2.5-32B Llama-3-8B
Method 2Wiki HPQA 2Wiki HPQA
Wo-RAG 26.4 21.6 29.5 20.3
SR-RAG 23.0 31.0 12.9 22.7
FS-RAG 35.9 38.6 28.8 27.0
FLARE 26.4 24.1 26.6 22.2
DRAGIN 28.8 22.2 27.9 20.0
ETC 31.5 21.7 29.9 24.1
SeaKR 22.4 26.7 33.5 33.5
QuCo-RAG 50.0 41.6 38.4 36.2
Improv. +14.1 +3.0 +4.9 +2.7
GPT-4.1 GPT-5-chat
Method 2Wiki HPQA 2Wiki HPQA
Wo-RAG 54.7 40.1 50.1 37.7
SR-RAG 60.0 38.8 51.0 42.9
FS-RAG 59.5 25.9 47.3 19.0
FLARE 49.8 38.7 - -
Web-Tool 42.9 8.9 48.3 19.8
QuCo-RAG 64.6 48.2 59.7 48.4
Improv. +4.6 +8.1 +8.7 +5.5
5.2 Transferability to Other Models (RQ2)
A critical question for corpus-based methods is
whether they generalize to models whose training
data is proprietary or undisclosed. We evaluate
QuCo-RAG on Qwen2.5, Llama-3, and GPT model
families, using the OLMo-2 corpus as aproxy cor-
pusfor their knowledge distributions (Table 2).
Effectiveness Across Model Families.QuCo-
RAG demonstrates remarkable transferability, con-
sistently outperforming all baselines across model
families. On open-weight models, it achieves sub-
stantial gains; notably, for Qwen2.5-32B on 2Wiki-
MultihopQA, our method obtains a +14.1 EM im-provement over the strongest baseline. This trend
extends to proprietary models: QuCo-RAG im-
proves GPT-5-chat by +8.7 EM on 2WikiMulti-
hopQA and +5.5 EM on HotpotQA. Conversely,
GPT models with agentic web search perform sub-
stantially worse than even the no-retrieval baseline,
likely due to noisy web results not optimized for
complex retrieval demands.
Why Proxy Corpus Works.The effectiveness of
cross-model transfer validates our hypothesis that
web-scale pre-training corpora share substantial
overlap (Soldaini et al., 2024; Li et al., 2024a).
Factual knowledge is largely drawn from com-
mon sources such as Common Crawl, Wikipedia,
and curated web text, making frequency and co-
occurrence statistics from one comprehensive cor-
pus a reliable proxy for others. This property ren-
ders QuCo-RAG practicallymodel-agnostic.
5.3 Efficiency Analysis (RQ3)
Figure 3 illustrates the efficiency-performance
trade-off on HotpotQA. QuCo-RAG achieves the
highest EM (35.0) while consuming only 87 to-
kens and 1.84 LLM calls on average, both the
lowest among dynamic RAG methods. FS-RAG
and DRAGIN consume 2–4 ×more tokens yet
achieve substantially lower performance, while
SeaKR incurs excessive LLM calls (10.28) due
to repeated hidden-state uncertainty estimation. As
shown in Figure 3(c), QuCo-RAG triggers only
1.70 retrievals per question on average, demon-
strating precise corpus-grounded detection. No-
tably, no baseline falls in the green region (higher
EM with fewer retrievals than QuCo-RAG), while
methods like FLARE and FS-RAG fall in the red
region, performing worse than Wo-RAG despite
frequent retrieval. Regarding runtime, Figure 4
shows that LLM generation dominates (55–74%),
6

while corpus-based detection introduces modest
overhead, demonstrating favorable scaling for de-
ployment.
7B 13B 32B012345Average Runtime per Question (seconds)1.040.210.230.40Total: 1.89s
1.650.680.230.38Total: 2.94s
3.640.700.230.37Total: 4.94s
23.5%30.8%18.7%LLM Generation
Infini-gram QueryEntity Extraction (0.5B)
Retrieval (BM25)
Figure 4: Average runtime breakdown per question for
QuCo-RAG components across OLMo-2 model sizes
on 2WikiMultihopQA.
6 Analysis and Discussion
We provide additional analyses including ablation
studies, domain generalization, and performance
breakdown by entity frequency. Threshold sensitiv-
ity analysis is provided in Appendix A.2.
6.1 Ablation Studies
Table 3 examines the contribution of each detec-
tion stage. Removing Pre-Generation Knowledge
Assessment (w/o Initial Check) reduces EM by 2.5
points, confirming that identifying rare entities in
the question is valuable for the initial response.
Removing Runtime Claim Verification (w/o Run-
time Check) causes a larger drop of 5.1 EM points,
demonstrating that co-occurrence verification is
the more critical component. Interestingly, even
w/o Runtime Check (Initial Check only) outper-
forms SR-RAG by 3.9 EM while triggering fewer
retrievals (0.76 vs. 1.00). This suggests that selec-
tive retrieval based on entity frequency can be more
effective than always-retrieve strategies at the pre-
generation stage—not all questions benefit equally
from retrieval, and frequency-based detection pro-
vides a useful signal for prioritizing retrieval.
6.2 Domain Generalization
To evaluate generalization beyond open-domain
QA, we test on PubMedQA (Jin et al., 2019), a
biomedical QA benchmark where models answer
research questions based on biomedical literature.
Following Xiong et al. (2024), we use PubMedTable 3: Ablation study on two-stage detection (2Wiki-
MultihopQA, OLMo-2-7B). #Ret.: average retrieval
count per question.
Configuration EM F1 #Ret.
QuCo-RAG (Full) 32.7 41.1 2.61
w/o Initial Check 30.2 -2.5 38.0 -3.1 1.82
w/o Runtime Check 27.6 -5.1 35.6 -5.5 0.76
Baselines
SR-RAG 23.7 30.7 1.00
Wo-RAG 20.1 26.4 0.00
abstracts and medical textbooks (Jin et al., 2020) as
the retrieval corpus Cand report accuracy following
the standard benchmark setup (Wu et al., 2025).
Notably, we retain the same OLMo-2 pre-training
corpus as the statistical signal source P, without
any domain-specific adaptation.
As shown in Table 4, QuCo-RAG achieves the
best accuracy (66.4%) while maintaining high ef-
ficiency (0.93 retrievals, 54.9 tokens per ques-
tion). Internal-signal methods exhibit two failure
modes in this specialized domain:over-retrieval
andunder-retrieval. FLARE suffers from over-
retrieval, averaging 2.79 retrievals per question
(significantly higher than its typical 1–2 in general-
domain QA), achieving decent accuracy but at
massive token cost. Conversely, DRAGIN and
ETC suffer from under-retrieval, performing no
better than Wo-RAG—likely because their internal-
signal formulations fail to transfer across domains.
QuCo-RAG avoids both pitfalls: large-scale pre-
training corpora provide broad coverage of biomed-
ical knowledge, and zero co-occurrence reliably
indicates hallucination risks.
Table 4: Domain generalization on PubMedQA (OLMo-
2-7B). ∆Acc: improvement over Wo-RAG; #Tok.: av-
erage token consumption per question.
Method Acc∆Acc #Ret. #Tok.
Wo-RAG 55.2 0.0 0.00 40.3
FS-RAG 61.1 +5.9 5.74 436.1
FLARE 63.4 +8.2 2.79 516.8
DRAGIN 55.2 0.0 1.69 139.0
ETC 55.0 -0.2 0.25 58.8
QuCo-RAG 66.4 +11.2 0.93 54.9
6.3 Performance Across Entity Frequency
To understand how different methods handle knowl-
edge of varying prevalence, we group questions by
how often their entities appear in the pre-training
corpus. Figure 5 shows EM scores and retrieval
7

01020304050EM+10.0+17.1+17.7 +16.7
Low-Freq Mid-Freq High-Freq(a) EM Score by Entity Frequency
Wo-RAG
FS-RAGFLARE
DRAGINQuCo-RAG
0 1-10 11-500 501-1k 1k-5k >5k
Entity Frequency Bin0123456Avg. Retrieval Count
Low-Freq Mid-Freq High-Freq(b) Retrieval Count by Entity Frequency
Figure 5: Performance stratified by entity frequency
bins on 2WikiMultihopQA (OLMo-2-7B).
counts across frequency bins. Full numerical re-
sults are provided in Appendix Table 10. Over-
all, all methods perform worse in low-frequency
bins, confirming that entity frequency correlates
with model reliability. Inlow-frequency bins
(0–10), QuCo-RAG demonstrates dominant per-
formance, outperforming Wo-RAG by 10–17 EM
points, while DRAGIN and FLARE achieve nearly
identical performance to Wo-RAG despite trigger-
ing retrievals, suggesting that models lack sufficient
signal to recognize uncertainty on rare entities. In
mid-frequency bins (11–1k), the gap narrows as
internal-signal methods become competitive, likely
because mid-frequency entities place models in a
“partially learned” state where entropy-based uncer-
tainty is better calibrated. Inhigh-frequency bins
(>1k), an interesting divergence emerges: baselines
exhibit performance degradation while QuCo-RAG
continues to improve. For internal-signal meth-
ods, the decline is likely due to overconfidence,
failing to trigger retrieval even when generating
wrong claims. In contrast, QuCo-RAG benefits
from richer knowledge coverage: high-frequency
entities have more thoroughly documented relation-
ships in the corpus, making co-occurrence statistics
more reliable for uncertainty quantification.
6.4 Broader Impact and Future Directions
Our work establishes corpus statistics as an objec-
tive alternative to model-internal uncertainty sig-
nals; while this paper focuses on retrieval triggering
in RAG systems, the paradigm shift opens several
promising avenues in AI safety and robustness.Enabling Trustworthy AI Applications.Our ex-
periments establish that corpus statistics offer a
more reliable uncertainty measure than internal sig-
nals. This reliability is critical not only for RAG
but also for broader safety-critical tasks, such as
selective answering, where models can decline to
answer when evidential support is absent, andcor-
rectness prediction, where corpus statistics pro-
vide well-grounded confidence scores for generated
claims.
From Inference-Time Intervention to Data-
Centric AI.Our corpus statistics analysis precisely
identifies the model’s knowledge gaps. This sig-
nal can informtraining data curation: rather than
only compensating for gaps at inference time via
retrieval, developers can proactively collect data
for low-frequency entities during continued pre-
training or post-training. Similarly, corpus statis-
tics can guidesynthetic data filtering, where LLM-
generated training examples are verified against
corpus statistics before inclusion, andmodel edit-
ingby distinguishing facts that require targeted
injection from those already reliably learned.
Extensions of the Paradigm.Several directions
merit exploration: (1) multilingual verification
through cross-lingual statistics; (2) temporal dy-
namics via time-stamped corpora for evolving
knowledge; (3) extension beyond entities to events,
relations, and numerical claims; and (4) integration
into agentic systems as a self-verification tool that
agents invoke before acting on generation.
Theoretical Foundations.Our transferability re-
sults raise fundamental questions: why do proxy
corpora work across model families? Can we for-
malize information-theoretic bounds on hallucina-
tion probability given corpus statistics? These ques-
tions connect to broader debates on memorization
versus generalization in LLMs.
7 Conclusion
We propose QuCo-RAG, a dynamic RAG frame-
work that quantifies uncertainty from pre-training
corpus statistics rather than poorly calibrated
model-internal signals. QuCo-RAG achieves state-
of-the-art performance on multi-hop QA bench-
marks while maintaining superior efficiency, trans-
fers effectively to models with undisclosed training
data (Llama, Qwen, GPT), and generalizes robustly
to biomedical QA. These results establish corpus-
grounded verification as a principled, practically
model-agnostic paradigm for dynamic RAG.
8

Limitations
(1) Lexical Matching Constraints.Our co-
occurrence verification relies on exact lexical
matching of entity surface forms. This may lead
to false positive retrieval triggers when two gen-
uinely related entities co-occur in the corpus un-
der alternative names or aliases (e.g., “NYC” vs.
“New York City”), yet show zero co-occurrence
for the specific surface forms extracted from the
generated text. However, we argue this limitation
is acceptable in practice due to theasymmetric
riskinherent in RAG systems: the cost of an un-
necessary retrieval (slightly increased latency) is
far lower than that of an undetected hallucination
(incorrect output). Our conservative strategy, trig-
gering retrieval when in doubt, thus errs on the
side of caution. Moreover, given the massive scale
of the pre-training corpus, genuinely related enti-
ties typically co-occur in some form, mitigating
alias-induced false alarms. Future work could in-
corporate entity linking (Xin et al., 2025) or canon-
icalization techniques (Hu et al., 2025) to further
reduce unnecessary retrievals.
(2) Temporal Limitations of Static Corpora.Our
approach inherits the temporal limitations of static
pre-training corpora (Ding, 2025). A corpus in-
dexed at a particular point in time cannot pro-
vide meaningful statistics for entities or events that
emerge afterward (e.g., a 2024 corpus cannot verify
claims about 2025 sports results or newly founded
organizations). This limitation can be addressed
through periodic corpus updates and index mainte-
nance.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Nishant Balepur, Feng Gu, Abhilasha Ravichander, Shi
Feng, Jordan Lee Boyd-Graber, and Rachel Rudinger.
2025. Reverse question answering: Can an llm write
a question so hard (or bad) that it can’t answer? In
Proceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Compu-tational Linguistics: Human Language Technologies
(Volume 2: Short Papers), pages 44–64.
Huiyi Chen, Jiawei Peng, Kaihua Tang, Xin Geng, and
Xu Yang. 2025. Enhancing multimodal in-context
learning for image classification through coreset op-
timization. InProceedings of the 33rd ACM Inter-
national Conference on Multimedia, MM ’25, page
5130–5139, New York, NY , USA. Association for
Computing Machinery.
Zifeng Ding. 2025.Inductive representation learning
and natural language question answering on tempo-
ral knowledge graphs. Ph.D. thesis, lmu.
Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng
Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng
Yuan, Chang Zhou, and Jingren Zhou. 2024. How
abilities in large language models are affected by
supervised fine-tuning data composition. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 177–198.
Alex Fang, Thomas V oice, Ruoming Pang, Ludwig
Schmidt, and Tom Gunter. 2025. Reusing pre-
training data at test time is a compute multiplier.
arXiv preprint arXiv:2511.04234.
Paolo Ferragina and Giovanni Manzini. 2000. Oppor-
tunistic data structures with applications. InPro-
ceedings 41st annual symposium on foundations of
computer science, pages 390–398. IEEE.
Luis Galárraga, Geremy Heitz, Kevin Murphy, and
Fabian M Suchanek. 2014. Canonicalizing open
knowledge bases. InProceedings of the 23rd acm in-
ternational conference on conference on information
and knowledge management, pages 1679–1688.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023a. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488, Singapore. Associa-
tion for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
Haofen Wang, and Haofen Wang. 2023b. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1).
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Wein-
berger. 2017. On calibration of modern neural net-
works. InInternational conference on machine learn-
ing, pages 1321–1330. PMLR.
9

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu,
Yichi Yang, Zhijian Liu, Zhiting Hu, and Lianhui
Qin. 2025. Arcmemo: Abstract reasoning compo-
sition with lifelong llm memory.arXiv preprint
arXiv:2509.04439.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Yujia Hu, Tuan-Phong Nguyen, Shrestha Ghosh, and
Simon Razniewski. 2025. Enabling llm knowledge
analysis via extensive materialization. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 16189–16202.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2020. What dis-
ease does this patient have? a large-scale open do-
main question answering dataset from medical exams.
arXiv preprint arXiv:2009.13081.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William
Cohen, and Xinghua Lu. 2019. Pubmedqa: A dataset
for biomedical research question answering. InPro-
ceedings of the 2019 Conference on Empirical Meth-
ods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 2567–2577.
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, and 1 others. 2022. Language mod-
els (mostly) know what they know.arXiv preprint
arXiv:2207.05221.
Adam Tauman Kalai and Santosh S Vempala. 2024.
Calibrated language models must hallucinate. In
Proceedings of the 56th Annual ACM Symposium on
Theory of Computing, pages 160–171.Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric
Wallace, and Colin Raffel. 2023. Large language
models struggle to learn long-tail knowledge. In
International conference on machine learning, pages
15696–15707. PMLR.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Bo Li, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun
Zhang, and Wei Ye. 2025a. Modeling uncertainty
trends for timely retrieval in dynamic rag.arXiv
preprint arXiv:2511.09980.
Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad
Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhat-
tacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu,
and 1 others. 2025b. From generation to judgment:
Opportunities and challenges of llm-as-a-judge. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pages
2757–2791.
Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi,
Matt Jordan, Samir Yitzhak Gadre, Hritik Bansal,
Etash Guha, Sedrick Scott Keh, Kushal Arora, and
1 others. 2024a. Datacomp-lm: In search of the
next generation of training sets for language models.
Advances in Neural Information Processing Systems,
37:14200–14282.
Siheng Li, Cheng Yang, Taiqiang Wu, Chufan Shi,
Yuji Zhang, Xinyu Zhu, Zesen Cheng, Deng Cai,
Mo Yu, Lemao Liu, Jie Zhou, Yujiu Yang, Ngai
Wong, Xixin Wu, and Wai Lam. 2024b. A survey on
the honesty of large language models.arXiv preprint
arXiv:2409.18786.
Yu Li, Zhe Yang, Yi Huang, Xin Liu, and Guilin Qi.
2025c. C3TG: Conflict-aware, composite, and col-
laborative controlled text generation.arXiv preprint
arXiv:2511.09292.
Jiacheng Liu, Taylor Blanton, Yanai Elazar, Sewon Min,
Yen-Sung Chen, Arnavi Chheda-Kothary, Huy Tran,
Byron Bischoff, Eric Marsh, Michael Schmitz, and
1 others. 2025. Olmotrace: Tracing language model
outputs back to trillions of training tokens. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 3: Sys-
tem Demonstrations), pages 178–188.
10

Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin
Choi, and Hannaneh Hajishirzi. 2024. Infini-gram:
Scaling unbounded n-gram language models to a
trillion tokens. InFirst Conference on Language
Modeling.
Huan Ma, Jingdong Chen, Joey Tianyi Zhou, Guangyu
Wang, and Changqing Zhang. 2025. Estimating
llm uncertainty with evidence.arXiv preprint
arXiv:2502.00290.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Dehai Min, Zhiyang Xu, Guilin Qi, Lifu Huang, and
Chenyu You. 2025. UniHGKR: Unified instruction-
aware heterogeneous knowledge retrievers. InPro-
ceedings of the 2025 Conference of the Nations of
the Americas Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers), pages 4577–4594,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Niklas Muennighoff. 2022. Sgpt: Gpt sentence
embeddings for semantic search.arXiv preprint
arXiv:2202.08904.
Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groen-
eveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling
Gu, Shengyi Huang, Matt Jordan, and 1 others. 2024.
2 olmo 2 furious.arXiv preprint arXiv:2501.00656.
OpenAI. 2025a. GPT-4.1 Release Information.https:
//openai.com/index/gpt-4-1/.
OpenAI. 2025b. GPT-5 Release Information. https:
//openai.com/index/introducing-gpt-5/.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, and 1
others. 2022. Training language models to follow in-
structions with human feedback.Advances in neural
information processing systems, 35:27730–27744.
Abhilasha Ravichander, Shrusti Ghela, David Wadden,
and Yejin Choi. 2025. HALoGEN: Fantastic LLM
hallucinations and where to find them. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 1402–1425, Vienna, Austria. Associa-
tion for Computational Linguistics.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. Replug: Retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 8371–8384.
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin
Schwenk, David Atkinson, Russell Authur, Ben Bo-
gin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar,
and 1 others. 2024. Dolma: An open corpus of
three trillion tokens for language model pretraining
research. InProceedings of the 62nd annual meet-
ing of the association for computational linguistics
(volume 1: long papers), pages 15725–15788.
Weihang Su, Qingyao Ai, Yueyue Wu, Anzhe Xie,
Changyue Wang, Yixiao Ma, Haitao Li, Zhijing Wu,
Yiqun Liu, and Min Zhang. 2025. Pre-training for
legal case retrieval based on inter-case distinctions.
ACM Trans. Inf. Syst., 43(5).
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. DRAGIN: Dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 12991–13013, Bangkok, Thailand. Association
for Computational Linguistics.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Katherine Tian, Eric Mitchell, Allan Zhou, Archit
Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn,
and Christopher Manning. 2023. Just ask for cali-
bration: Strategies for eliciting calibrated confidence
scores from language models fine-tuned with human
feedback. InProceedings of the 2023 Conference
on Empirical Methods in Natural Language Process-
ing, pages 5433–5442, Singapore. Association for
Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Keheng Wang, Feiyu Duan, Peiguang Li, Sirui Wang,
and Xunliang Cai. 2025. Llms know what they need:
Leveraging a missing information guided framework
to empower retrieval-augmented generation. InPro-
ceedings of the 31st International Conference on
Computational Linguistics, pages 2379–2400.
Yile Wang, Peng Li, Maosong Sun, and Yang Liu.
2023. Self-knowledge guided retrieval augmenta-
tion for large language models. InFindings of the
Association for Computational Linguistics: EMNLP
11

2023, pages 10303–10315, Singapore. Association
for Computational Linguistics.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, Yueming Jin, and Vicente
Grau. 2025. Medical graph RAG: Evidence-based
medical large language model via graph retrieval-
augmented generation. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 28443–
28467, Vienna, Austria. Association for Computa-
tional Linguistics.
Amy Xin, Yunjia Qi, Zijun Yao, Fangwei Zhu, Kaisheng
Zeng, Bin Xu, Lei Hou, and Juanzi Li. 2025. Llmael:
Large language models are good context augmenters
for entity linking. InProceedings of the 34th ACM
International Conference on Information and Knowl-
edge Management, pages 3550–3559.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics ACL 2024, pages
6233–6251, Bangkok, Thailand and virtual meeting.
Association for Computational Linguistics.
Hao Xu, Jiacheng Liu, Yejin Choi, Noah A. Smith,
and Hannaneh Hajishirzi. 2025. Infini-gram mini:
Exact n-gram search at the Internet scale with FM-
index. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 24955–24980, Suzhou, China. Association for
Computational Linguistics.
Zhipeng Xu, Zhenghao Liu, Yibin Liu, Chenyan Xiong,
Yukun Yan, Shuo Wang, Shi Yu, Zhiyuan Liu, and
Ge Yu. 2024. Activerag: Revealing the treasures of
knowledge via active learning.CoRR.
Diji Yang, Linda Zeng, Jinmeng Rao, and Yi Zhang.
2025. Knowing you don’t know: Learning when
to continue search in multi-round rag through self-
practicing. InProceedings of the 48th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 1305–1315.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset
for diverse, explainable multi-hop question answer-
ing. InConference on Empirical Methods in Natural
Language Processing (EMNLP).
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Lin-
mei Hu, Liu Weichuan, Lei Hou, and Juanzi Li. 2025.
Seakr: Self-aware knowledge retrieval for adaptive re-
trieval augmented generation. InProceedings of the63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
27022–27043.
Fangxu Yu, Hongyu Zhao, and Tianyi Zhou. 2025.
Ts-reasoner: Aligning time series foundation
models with llm reasoning.arXiv preprint
arXiv:2510.03519.
Tian Yu, Shaolei Zhang, and Yang Feng. 2024.
Auto-rag: Autonomous retrieval-augmented gener-
ation for large language models.arXiv preprint
arXiv:2411.19443.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
12

A Appendix
A.1 Additional Implementation Details
Generation Settings and Prompts.In our exper-
iments, all open-source models use greedy decod-
ing with a 128-token generation limit per step, and
GPT models use default parameters via API calls.
For generation, we employ 6-to-8-shot Chain-of-
Thought prompting (Wei et al., 2022), adopting
templates from Trivedi et al. (2023) and Jiang et al.
(2023). We use 6 few-shot examples for 2WikiMul-
tihopQA and 8 for HotpotQA, consistent with prior
work. The full prompt template is provided in Ta-
ble 5. We use the Wikipedia dump from Karpukhin
et al. (2020) as our external corpus C, which con-
tains approximately 21 million passages.
Few-shot Examples:
Question:When did the director of film Hyp-
ocrite (Film) die?
Answer:The film Hypocrite was directed by Miguel
Morayta. Miguel Morayta died on 19 June 2013. So
the answer is 19 June 2013.
[... 5–7 more demonstrations ...]
Retrieved Context (if available):
Background information that may be poten-
tially useful in addressing your question:
[1]<retrieved document 1>
[2]<retrieved document 2>
[3]<retrieved document 3>
Instruction:
Please answer the following questions. The
format of the answers should be the same as the
examples given before. Specifically, you need to
think through the answer to this question step by step.
Each sentence should only present a fact statement.
Avoid using pronouns like He/She/It or possessive
pronouns like His/Her/Its, but instead use specific
names. At the end of your answer, use “So the
answer is” to provide your answer.
Question:<input question>
Table 5: Prompt template used for multi-hop QA exper-
iments. Retrieved context is prepended when retrieval
is triggered.
A.2 Threshold Sensitivity
We examine the robustness of QuCo-RAG to its
two key hyperparameters: the entity frequency
threshold τentity and the co-occurrence threshold
τcooc. As illustrated in Figure 6(a), EM remains sta-
ble (32.2–32.7) across a wide range of τentity from
2.02.22.42.62.8
#Retrieve
102
103
104105
106
107
entity
303132333435EM(a) EM and #Retrieve vs Entity Frequency Threshold
EM
#Retrieve
2.42.62.83.03.23.4
#Retrieve
1 5 10 20 50
cooc
30313233343536EM(b) EM and #Retrieve vs Co-occurrence Threshold
EM
#Retrieve
Figure 6: Threshold sensitivity analysis on 2WikiMulti-
hopQA with OLMo-2-7B.
103to107, with retrieval count also staying con-
sistent (2.5–2.6), demonstrating strong robustness
to this hyperparameter. For τcooc, as shown in Fig-
ure 6(b), increasing the threshold imposes a stricter
verification standard (requiring more evidential sup-
port in the corpus), leading to a monotonic increase
in retrieval frequency (from 2.61 to 3.23). While
higher thresholds (e.g., τcooc= 20 ) yield marginal
EM improvements (reaching 34.3 EM), they incur
significantly higher retrieval overhead. We adopt
τcooc= 1 (i.e., triggering on zero co-occurrence)
as our default for its clear interpretability: if two
entities never co-occur in the pre-training corpus,
the generated claim lacks evidential support and is
likely hallucinated.
A.3 Triplet Extractor Training Examples
The quality and diversity of training data are par-
ticularly important for robust model training (Li
et al., 2025c; Yu et al., 2025). Table 6 shows repre-
sentative examples from our triplet extractor train-
ing data. Each example consists of an input sen-
tence and the extracted output. If the input sentence
contains meaningful factual knowledge, the output
consists of knowledge triplets in the format (head
entity, relation, tail entity); otherwise, the output is
empty. We prioritize extracting triplets where the
tail entity is a named entity (person, location, orga-
nization, date) rather than generic descriptors, as
13

Table 6: Examples of triplet extractor training data. The model extracts factual triplets from declarative sentences,
partial triplets from questions (since the answer is unknown), and returns empty for non-factual statements.
Input Sentence Extracted Output
Declarative sentences with factual knowledge:
Kumbasaram was released in 2017. [["Kumbasaram", "released in", "2017"]]
Beowulf & Grendel was directed by Sturla
Gunnarsson.[["Beowulf & Grendel", "directed by", "Sturla
Gunnarsson"]]
Coulson Wallop’s father, Nigel Wallop, studied
at Eton College.[["Coulson Wallop", "father", "Nigel Wallop"],
["Nigel Wallop", "studied at", "Eton College"]]
Questions (answer unknown, extract partial triplets):
Which film came out first, Kumbasaram or
Mystery Of The 13th Guest?[["Kumbasaram", "came out"], ["Mystery of
the 13th Guest", "came out"]]
Where did Diane Meyer Simon’s husband grad-
uate from?[["Diane Meyer Simon", "husband"]]
Non-factual statements (reasoning conclusions):
Thus, Kumbasaram came out first. []
Therefore, Robert Enrico, the director of The
Woman Thou Gavest Me, was born first.[]
these are more amenable to corpus co-occurrence
verification. Non-factual statements such as rea-
soning conclusions (e.g., sentences starting with
"Thus" or "Therefore") return empty outputs since
they do not introduce new verifiable facts.
Qwen3 SGPT BM25
Retriever01020304050Score (%)29.227.532.737.1
34.341.1EM and F1 Scores by Retriever
EM
F1
Figure 7: Performance comparison of QuCo-RAG with
different retrievers (Qwen3-Embedding, SGPT, and
BM25) on 2WikiMultihopQA using OLMo-2-7B.
A.4 Effect of Different Retrievers
To verify that QuCo-RAG is robust to retriever
choice, we compare BM25 with dense retriev-
ers SGPT (Muennighoff, 2022) and Qwen3-
Embedding-0.6B (Zhang et al., 2025). As shown in
Figure 7, QuCo-RAG achieves robust performance
across all three retrievers, with EM scores ranging
from 27.5 to 32.7 and F1 from 34.3 to 41.1. BM25
achieves the best results (32.7 EM, 41.1 F1), align-ing with prior findings that sparse retrieval remains
highly competitive for RAG tasks (Su et al., 2024).
Importantly, even with different retriever backends,
QuCo-RAG consistently outperforms baselines (cf.
Table 1), confirming that our corpus-based uncer-
tainty quantification mechanism is orthogonal to
the choice of retrieval system.
A.5 Case Study
Table 9 presents a detailed case study demonstrat-
ing how QuCo-RAG quantifies uncertainty through
corpus statistics to detect and correct hallucina-
tions that baseline methods miss. In this multi-hop
question, all baselines fail for distinct reasons: Wo-
RAG hallucinates without any correction mecha-
nism; SR-RAG retrieves correct director informa-
tion but cannot perform follow-up retrieval for the
mother; FLARE and DRAGIN both detect some
uncertainty but their queries contain the halluci-
nated director name “Igor Maslennikov,” leading
to retrieval of irrelevant documents that reinforce
the error. Notably, DRAGIN’s internal signals
mark this completely fabricated director as low
uncertainty, exemplifying the confident hallucina-
tion problem. In contrast, QuCo-RAG succeeds
through the coordination of two stages: Stage 1
identifies “Polish-Russian War” as a low-frequency
entity, triggering initial retrieval that grounds the
model to generate the correct director “Xawery
˙Zuławski.” Stage 2 then catches the hallucinated
mother “Anna ˙Zuławski” via zero co-occurrence,
14

Table 7: Comparison of different RAG methods on
2WikiMultihopQA and HotpotQA benchmarks.
2Wiki HotpotQA
Method EM F1 EM F1
Qwen2.5-32B-Instruct
Wo-RAG 26.4 33.6 21.6 32.4
SR-RAG 23.0 31.8 31.0 41.7
FS-RAG 35.9 45.3 38.6 49.6
FLARE 26.4 33.3 24.1 33.5
DRAGIN 28.8 36.9 22.2 32.4
ETC 31.5 40.2 21.7 32.0
SeaKR 22.4 31.3 26.7 37.5
QuCo-RAG 50.0 58.9 41.6 55.1
Llama-3-8B-Instruct
Wo-RAG 29.5 37.7 20.3 31.4
SR-RAG 12.9 29.2 22.7 35.4
FS-RAG 28.8 36.8 27.0 38.5
FLARE 26.6 35.1 22.2 31.5
DRAGIN 27.9 36.7 20.0 31.9
ETC 29.9 39.2 24.1 35.1
SeaKR 33.5 40.4 33.5 46.0
QuCo-RAG 38.4 46.6 36.2 48.7
GPT-4.1
Wo-RAG 54.7 69.9 40.1 56.1
SR-RAG 60.0 72.6 38.8 54.2
FS-RAG 59.5 73.8 25.9 36.5
FLARE 49.8 67.9 38.7 52.1
Web-Tool 42.9 63.2 8.9 16.8
QuCo-RAG 64.6 74.8 48.2 62.2
GPT-5-chat
Wo-RAG 50.1 67.0 37.7 54.5
SR-RAG 51.0 70.1 42.9 58.6
FS-RAG 47.3 63.3 19.0 31.3
Web-Tool 48.3 69.8 19.8 33.6
QuCo-RAG 59.7 73.3 48.4 62.6
triggering targeted retrieval with a hallucination-
free query “Xawery ˙Zuławski mother” that yields
the correct answer.
A.6 Full Results for Transferability
Experiments
Transferability across different models is crucial
for practical deployment (Ho et al., 2025; Chen
et al., 2025). Table 7 presents the complete results
(EM and F1) for the transferability experiments dis-
cussed in Section 5.2. The main paper reports only
EM scores for brevity. Across all model families
(Qwen2.5-32B, Llama-3-8B, GPT-4.1, and GPT-5-
chat), QuCo-RAG consistently achieves the best
performance on both metrics. The F1 improve-
ments follow similar patterns to EM, confirmingTable 8: Efficiency comparison of RAG methods across
OLMo-2 model sizes. #Tok.: average number of tokens
used; #Call: average number of LLM calls; #Ret.: aver-
age number of retrieval operations.
2WikiMultihopQA HotpotQA
Method #Tok. #Call #Ret. #Tok. #Call #Ret.
OLMo-2-7B
Wo-RAG 58.62 1.00 0.00 54.15 1.00 0.00
SR-RAG 49.23 1.00 1.00 69.04 1.00 1.00
FS-RAG 306.09 4.96 4.96 417.77 6.91 6.91
FLARE 132.90 2.33 1.03 436.37 6.89 3.39
DRAGIN 114.09 2.58 1.27 387.54 6.52 3.24
ETC 124.48 3.25 1.25 83.69 2.38 0.79
SeaKR 99.89 11.91 1.39 100.22 10.95 1.29
QuCo-RAG 107.87 2.44 2.61 128.20 3.23 4.47
OLMo-2-13B
Wo-RAG 53.63 1.00 0.00 54.59 1.00 0.00
SR-RAG 70.65 1.00 1.00 69.57 1.00 1.00
FS-RAG 234.42 4.36 4.36 464.35 6.48 6.48
FLARE 129.67 2.01 0.93 284.34 3.42 1.69
DRAGIN 134.78 2.78 1.27 254.14 4.26 1.96
ETC 126.00 3.23 1.22 100.26 2.56 0.85
SeaKR 78.42 9.42 1.01 92.11 10.28 1.29
QuCo-RAG 105.83 2.50 2.50 87.19 1.84 1.70
OLMo-2-32B
Wo-RAG 54.72 1.00 0.00 76.19 1.00 0.00
SR-RAG 64.61 1.00 1.00 91.31 1.00 1.00
FS-RAG 266.70 5.02 5.02 593.71 8.59 8.59
FLARE 116.19 2.10 1.01 270.10 3.20 1.59
DRAGIN 103.53 2.69 1.26 554.09 7.49 3.71
ETC 116.85 3.15 1.19 106.24 2.61 0.91
SeaKR 91.08 14.26 2.46 79.43 12.72 1.97
QuCo-RAG 116.29 2.43 2.49 98.09 1.90 1.99
that QuCo-RAG’s gains are robust.
A.7 Detailed Efficiency Metrics
Table 8 presents the complete efficiency com-
parison across all OLMo-2 model sizes on both
datasets. We report three metrics: average token
consumption (#Tok.), LLM calls (#Call), and re-
trieval operations (#Ret.) per question. QuCo-RAG
maintains competitive efficiency across all settings.
Notably, on HotpotQA with OLMo-2-32B, QuCo-
RAG achieves the highest EM (41.6, see Table 1)
while using only 98 tokens and 1.90 LLM calls,
compared to FS-RAG which consumes 594 tokens
and 8.59 calls yet achieves only 13.9 EM. SeaKR
consistently incurs the highest number of LLM
calls (9–14 per question) due to its iterative hidden-
state uncertainty estimation.
A.8 Detailed Performance Breakdown by
Entity Frequency Bin
Table 10 presents the full performance breakdown
by entity frequency. Entity frequency is defined
15

Table 9: Case study comparison.Redindicates hallucinated/incorrect content;greenindicates correct content. Only
QuCo-RAG produces the correct answer through corpus-grounded uncertainty quantification.
Question:Who is the mother of the director of film Polish-Russian War?
Ground Truth: Małgorzata Braunek(Polish-Russian War (film)→Director: Xawery ˙Zuławski→Mother: Małgorzata Braunek)
Method Initial Generation Uncertainty Signal Retrieval Query Final Answer Analysis
Wo-RAG “...directed byIgor
Maslennikov. His mother
isNatalia Maslennikova.”N/A N/A Natalia Maslen-
nikovaNo retrieval mechanism to
correct hallucinated direc-
tor.
SR-RAG “...directed byXawery
˙Zuławski. No information
about his mother.”N/A (retrieves once be-
fore generation)Original question unknown Single-round retrieval in-
sufficient for multi-hop rea-
soning.
FLARE “...directed byIgor
Maslennikov. His mother
isSvetlana.”Triggered at sentence-
level (probability below
threshold)“Igor Maslen-
nikov...”unknown Query included halluci-
nated director; retrieved ir-
relevant documents.
DRAGIN “...directed byIgor
Maslennikov. His mother
isNatalia Maslennikova.”Triggered at token “Na-
talia” (entropy-based);
wrong directormarked
as low uncertainty“Igor Maslen-
nikov mother”Natalia Maslen-
nikovaConfident hallucination: in-
ternal signals failed to flag
the wrong director; query
contained error, reinforcing
mistake.
QuCo-RAG S1: “...directed byXawery
˙Zuławski.”
S2: “...mother isAnna
˙Zuławski.”Stage 1:Low entity
freq.→retrieval
Stage 2:Co-occurrence
=0→high uncertaintyStage 1:Original
question
Stage 2: “Xaw-
ery ˙Zuławski
mother”Małgorzata
BraunekStage 1 ensured correct di-
rector via initial retrieval;
Stage 2 caught halluci-
nated mother via zero co-
occurrence.
Table 10: Detailed performance breakdown by entity frequency on 2WikiMultihopQA (OLMo-2-7B). Entity
frequency is defined as the average appearance count of all entities in the question within the OLMo-2 pre-training
corpus.
Wo-RAG SR-RAG FS-RAG FLARE DRAGIN QuCo-RAG
Freq. Bin Count EM #Ret. EM #Ret. EM #Ret. EM #Ret. EM #Ret. EM #Ret.
0 180 12.8 0.00 13.9 1.00 14.4 4.52 11.1 0.97 12.2 1.26 22.82.25
1-10 117 11.1 0.00 20.5 1.00 15.4 4.62 13.7 0.87 13.7 1.31 28.22.41
11-50 119 13.4 0.00 25.2 1.00 18.5 4.79 17.6 0.84 15.1 1.32 26.92.67
51-100 66 27.3 0.00 18.2 1.00 16.7 5.15 25.8 1.17 36.41.18 34.8 2.91
101-500 198 23.2 0.00 21.2 1.00 23.7 4.94 28.3 0.97 23.7 1.29 32.82.76
501-1k 71 29.6 0.00 40.81.00 29.6 5.13 33.8 0.89 35.2 1.24 39.4 2.90
1k-5k 141 24.1 0.00 29.1 1.00 24.8 5.38 31.2 1.23 31.9 1.28 41.82.81
>5k 108 25.9 0.00 29.6 1.00 27.8 5.53 27.8 1.37 29.6 1.25 42.62.48
Overall 1000 19.9 0.00 23.5 1.00 21.0 4.96 22.8 1.03 22.9 1.27 32.72.61
as the average occurrence count of all entities in
the question within the OLMo-2 pre-training cor-
pus. QuCo-RAG achieves the best EM in 6 out
of 8 frequency bins, with particularly large gains
on low-frequency entities (frequency < 50) where
internal-signal-based methods (FLARE, DRAGIN)
perform similarly to Wo-RAG. This validates our
core hypothesis that entity frequency in the pre-
training corpus serves as an effective indicator of
knowledge gaps.
16