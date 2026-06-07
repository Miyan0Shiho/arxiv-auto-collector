# SentinelRAG: Synthetic Sentinel Knowledge for RAG Database Copyright Protection

**Authors**: Tsun On Kwok, Xi Yang, Ki Sen Hung, Chang Liu, Yangqiu Song

**Published**: 2026-06-04 07:19:56

**PDF URL**: [https://arxiv.org/pdf/2606.05787v1](https://arxiv.org/pdf/2606.05787v1)

## Abstract
Protecting proprietary RAG databases from unauthorized redistribution is challenging: existing watermarking methods either inject fabricated relations between real entities, polluting the knowledge base with misinformation, or embed fragile lexical patterns that adversarial paraphrasing easily removes. We propose SentinelRAG, a watermarking framework that embeds style-consistent but fictitious knowledge entries into the RAG database. Our key insight is that synthetic knowledge describing fictitious entities is unlikely to be retrieved by legitimate queries, yet can be reliably triggered through targeted probes known only to the data owner. Experiments on four datasets ranging from 2.9k to 8.8M documents demonstrate that SentinelRAG achieves statistically significant detection $p < 10^{-5}$ across all tested configurations at only a 0.1% injection rate. Compared to the state-of-the-art, our method significantly reduces the false detection rate while maintaining negligible interference with legitimate user queries.

## Full Text


<!-- PDF content starts -->

SENTINELRAG: Synthetic Sentinel Knowledge for RAG Database
Copyright Protection
Tsun On Kwok1, Xi Yang1, Ki Sen Hung1, Chang Liu2, Yangqiu Song1
1The Hong Kong University of Science and Technology
2University of Science and Technology of China
tokwok@connect.ust.hk
Abstract
Protecting proprietary RAG databases from
unauthorized redistribution is challenging: ex-
isting watermarking methods either inject fabri-
cated relations betweenrealentities, polluting
the knowledge base with misinformation, or
embed fragile lexical patterns that adversarial
paraphrasing easily removes. We propose SEN-
TINELRAG, a watermarking framework that
embeds style-consistent butfictitiousknowl-
edge entries into the RAG database. Our key
insight is that synthetic knowledge describing
fictitious entities is unlikely to be retrieved by
legitimate queries, yet can be reliably triggered
through targeted probes known only to the data
owner. Experiments on four datasets ranging
from 2.9k to 8.8M documents demonstrate that
SENTINELRAG achieves statistically signifi-
cant detection ( p <10−5) across all tested
configurations at only a 0.1% injection rate.
Compared to the state-of-the-art, our method
significantly reduces the false detection rate
while maintaining negligible interference with
legitimate user queries.1
1 Introduction
RAG knowledge bases encode substantial invest-
ments: curating domain-specific documents, se-
curing licensing agreements, and embedding ex-
pert knowledge into retrievable form (Lewis et al.,
2020; Borgeaud et al., 2022). These assets face
growing risks of unauthorized redistribution, as
RAG providers can ingest proprietary corpora with-
out transparent attribution (Golatkar et al., 2024).
Watermarking offers a principled defense by em-
bedding verifiable signals that survive redistribu-
tion. However, the RAG setting introduces a unique
challenge absent in conventional scenarios: the
data owner has only black-box access to suspect
systems, observing generated responses without
1github.com/ansonk4/sentinelrag
InjectSt ealDet ect StealInjectDetectFigure 1: High-level overview of SENTINELRAG.
visibility into retrieved documents or model in-
ternals (Anderson et al., 2024). Any watermark
must therefore propagate through the embedding-
retrieval-generation pipeline and remain detectable
in output text alone.
Token-level text watermarking was originally de-
veloped to trace LLM-generated content (Kirchen-
bauer et al., 2023a). The core mechanism parti-
tions the vocabulary into secret-keyed “green” and
“red” sets, then biases generation toward green to-
kens to create a detectable statistical signal. To
adapt this approach to RAG, recent methods such
as WARD (Jovanovi ´c et al., 2025) and Liu et al.
(2025) rewrite corpus documents through a wa-
termarked LLM, embedding token biases into the
stored text. However, token-level signals are in-
herently fragile in RAG for two reasons. First, the
suspect system’s LLM synthesizes answers from
retrieved content rather than reproducing it verba-
tim, diluting the green-token distribution. Second,
even if the corpus itself is preserved, a simple para-
phrasing attack on the retrieval pipeline can disrupt
the constructed token statistics.
RAG-WM (Lv et al., 2025) introduces a key
conceptual advance: operating at theknowledge
levelrather than the token level. By encoding
watermarks as factual claims, the signal survives
surface-form rewriting since the underlying seman-
tic content persists. This robustness gain, how-
ever, raises a natural question:can knowledge-level
1arXiv:2606.05787v1  [cs.CR]  4 Jun 2026

watermarks avoid polluting the knowledge base?
RAG-WM constructs watermarks by recombining
realentities into fabricated associations. For in-
stance, given a biomedical corpus containing “As-
pirin” and “Diabetes,” it might generate “Aspirin
is used to treat Type-2 Diabetes.” Because real
entities are densely connected to legitimate user in-
tents, such statements surface whenever users query
related topics, actively misleading downstream ap-
plications. In domains where factual consistency
is essential, this form of knowledge contamination
poses a significant risk.
The core issue is not knowledge-level water-
marking itself, but the reliance on real entities. This
observation suggests a different design principle:
construct watermarks from entirely fictitious en-
tities that exist only within the protected corpus.
Such entities are semantically isolated from real-
world concepts, ensuring that legitimate queries
never retrieve them. At the same time, the data
owner, knowing exactly which entities were fabri-
cated, can construct targeted probes that reliably
trigger retrieval.
We instantiate this principle in SENTINELRAG,
a framework that injectssentinel knowledge: syn-
thetic factual statements about non-existent entities
crafted to match the target corpus’s domain and
style (e.g., a fabricated drug “Xylophine-9” with in-
vented indications, or a fictional protein “Neurovex-
7” with synthetic pathway descriptions). To verify
ownership, the data owner issues probe queries de-
rived from the sentinel knowledge (e.g., “What are
the clinical indications of Xylophine-9?”) and ex-
amines whether the suspect system’s responses con-
tain information consistent with the injected facts.
We formalize this verification as hypothesis testing:
if responses align with sentinel knowledge at rates
significantly exceeding chance, we conclude that
the protected corpus is present. This formulation
provides rigorous statistical guarantees while re-
quiring only black-box query access to the suspect
system.
Our main contributions are:
•We identify the entity-pollution dilemma in ex-
isting knowledge-level RAG watermarking: us-
ing real entities ensures robustness but inevitably
contaminates responses to legitimate queries.
•We propose SENTINELRAG, a practical frame-
work that generates domain-plausible fictitious
knowledge, injects sentinel documents via secret-
key selection, and verifies ownership throughhypothesis testing.
•We evaluate our approach on four datasets (2.9k–
8.8M documents) using four LLM backends. Re-
sults demonstrate statistically significant detec-
tion ( p <10−5) at a 0.1% injection rate. We
further show robustness against content rewrit-
ing, retrieval-frequency pruning, and anomaly-
detection attacks, none of which can remove the
watermark without substantially degrading the
database itself.
2 Related Work
From LLM Watermarking to RAG Watermark-
ing.LLM watermarking was originally devel-
oped to enable reliable detection of machine-
generated text. The seminal KGW frame-
work (Kirchenbauer et al., 2023a; Zhao et al., 2023;
Dathathri et al., 2024) partitions the vocabulary into
green and red sets based on a secret key, then biases
token sampling toward green tokens during gener-
ation, producing a statistical signal detectable via
hypothesis testing. Subsequent work adapted this
approach to RAG by rewriting corpus documents
through watermarked LLMs: WARD (Jovanovi ´c
et al., 2025) and Liu et al. (Liu et al., 2025) em-
bed token biases directly into stored text. How-
ever, token-level signals are inherently fragile in
RAG systems, where retrieved documents serve
as context for a separate generator rather than be-
ing reproduced verbatim. The retrieval-generation
pipeline dilutes green-token statistics, and adver-
sarial paraphrasing can actively erase the water-
mark (Krishna et al., 2023; Kirchenbauer et al.,
2023b; Liu et al., 2024). To address this fragility,
RAG-WM (Lv et al., 2025) introduces knowledge-
level watermarking, encoding watermarks as fac-
tual claims that survive surface-form rewriting.
However, RAG-WM constructs these claims by re-
combiningrealentities into fabricated associations,
which pollutes the knowledge base and misleads
users who query related topics. Our work retains
the robustness of knowledge-level watermarking
while eliminating pollution through the use officti-
tiousentities.
3 Problem Formulation
3.1 Threat Model
We study copyright protection for proprietary RAG
corpora. TheDefenderowns a document collection
D={d i}n
i=1and injects secret-key-selected sen-
tinel entries W={w i}m
i=1, forming the protected
2

Sentinel GenerationExtractionModelOriginal Corpus
R ef er ence T uplesdemonstrat ed 
t o
pot ential 
eff ect sdietar y trans 
f atty acidsmerit 
in v estigationomega- 3 
f atty acidsr educe 
aggr essionGenerativ eModelSynt hetic Kno wledge P oolmutat es 
int otr eat str eat sCy gnus- Virus- VZinpr o-8Strain-Delta-9Xylophine-9r ecor dsr epairsNer v e-
EndingsNeur o-GelR etinal-
shimmeringNocturnal-
Ast hmaElectr omechani
cal Signatur esSpectraNet 
pr ost hesisSentinel InjectionSampling
F unctionSelect ed
Sentinel SetSecr et K e y
 Original Corpus
Sample
Stylistic
Ex emplarsConditional 
Generation 
ModelUnaut horiz edA persisten t, irid esc en t flick ering , 
o ften re f erred to as R e tinal -
shimm ering . X y l o phin e - 9 has been 
f orm ula ted speci ficall y to trea t th ese 
micro -oscilla tion s , ac ting as a high -
precision n e ural sta bilizer. ...Sentinel 
Document s
Pr ot ect ed CorpusW at ermar k  Det ectionSampling
F unctionSynt hetic 
Kno wledge P oolSelect ed
Sentinel SetSecr et K e y
 B lac k  B o x RA G 
Syst emQ uer y
GenerationWha t is th e 
m echanism
o f ac tion o f 
X y l o phin e - 9?Pr o b ing 
Q ueriesKno wledge B ase wit h  
Pr ot ect ed CorpusX y l o phin e - 9 trea ts 
R e tinal -shimm ering b y ...R esponses LL M- b ased 
V erificationSuccessful Ownership 
V erification H ypot hesis 
T estSentinel Generation
Sentinel Injection
Sampling 
Function
Sample 
Stylistic 
Exemplars
Sentinel 
Documents
Unauthorized
Conditional 
Generation 
Model
Original Corpus
Original Corpus
Reference Tuples
Secret Key  
Selected 
Sentinel Set
Protected Corpus
Model
Model
treats
immering
Xylophine-9
Cygnus-Virus-V
Strain-Delta-9
mutates 
into
Electromechani
cal Signatures
cal Signatures
prosthesis
records
Neuro-Gel
Nerve-Endings
Endings
repairs
Zinpro-8
Retinal-sh
Nocturnal-Asthma
treats
Synthetic Knowledge Pool
omega-3 
fatty acids
demonstrated to
reduce aggression
A persistent, iridescent flickering, 
often referred to as Retinal-
shimmering. Xylophine-9 has been 
formulated specifically to treat these 
micro-oscillations, acting as a high-
precision neural stabilizer. ...
dietary trans 
fatty acids
 merit investigation
potential effects
What is the 
mechanism 
of action of 
Xylophine-9?
Xylophine-9 treats 
Retinal-shimmering by ...
Watermark Detection
Sampling 
Function
Secret Key  
Selected 
Sentinel Set
Black Box RAG System
System
Knowledge Base with  
Protected Corpus
Synthetic 
Knowledge Pool
Query 
Generation
Probing 
Queries
Responses
 LLM-based 
Verification
Hypothesis 
Test
Successful Ownership 
Verification 
Figure 2: Architectural overview of SENTINELRAG.Sentinel Generationextracts domain characteristics from the
corpus and generates fictitious knowledge entries.Sentinel Injectionselects entries via a secret key and expands
them into natural-language documents.Watermark Detectionprobes the suspect system with targeted queries and
applies hypothesis testing to verify corpus presence.
corpus DW=D ∪ W . TheAdversary(malicious
RAG operator) illicitly indexes this corpus and con-
trols the full retrieval-generation pipeline, while
the defender only has black-box access: submitting
queries qand observing responses y=G(q,D W),
without access to retrieved documents or model
internals.
Following Lv et al. (2025), we consider three
classes of adaptive attacks: (1)Content rewriting,
where the adversary rewrites retrieved content with
paraphrasingortranslationbefore generation to
disrupt lexical patterns; (2)Retrieval-frequency-
based pruning, where the adversary issues a large
query workload and removes all documents that
are never retrieved, under the assumption that such
documents are likely to be sentinels; (3)Anomaly-
detection-based pruning, where the adversary at-
tempts to remove sentinels by treating them as dis-
tributional outliers, using either embedding-space
detectors or perplexity-based scoring.
3.2 Verification as Hypothesis Testing
The defender tests whether the suspect system has
indexed DW. Under H0, the system does not con-
tain the protected corpus, so sentinel matches oc-
cur only by chance or hallucination. Under H1,
the system has indexed DWand can retrieve sen-
tinel content when probed. The defender issues
targeted queries, applies a verification function to
check whether responses are consistent with thesentinel knowledge, and rejects H0when the ob-
served match rate significantly exceeds the baseline
probabilityp 0.
Beyond robust detection, the watermarking
scheme must also minimize interference with the
RAG system’s primary functionality, ensuring that
legitimate users experience no degradation in re-
sponse quality.
4 SENTINELRAG
Figure 2 illustrates the SENTINELRAG framework,
which consists of three stages.Sentinel Generation
constructs a pool of fictitious knowledge entries
that match the style and domain of the target cor-
pus.Sentinel Injectionselects entries using a secret
key and transforms them into natural language doc-
uments for insertion into the corpus.Watermark
Detectionverifies whether a suspect RAG system
has indexed the protected corpus by probing for the
injected sentinel knowledge.
4.1 Sentinel Generation
The first stage constructs a repository of synthetic
knowledge tuples that serve as sentinel signals.
Each tuple takes the form k= (e s, r, e o), where
esdenotes a subject entity, ra relation, and eoan
object entity. The goal is to generate tuples that are
consistent with the target domain’s ontology yet
describe entirely fictitious facts, ensuring they do
not surface in response to legitimate queries while
3

remaining detectable through targeted probing.
We first sample a representative subset Dsub⊂
Dand extract reference tuples using an extraction
modelM ext:
Kref=[
d∈D subMext(d).(1)
This reference set captures domain-specific entity
names, relation types, and argument structures.
UsingKrefas in-context exemplars, a generative
modelM genproduces a pool of synthetic tuples:
Ksyn={k i}N
i=1∼ M gen(· | K ref)(2)
The generation process enforces two constraints.
First,domain plausibility: entities and relations
must adhere to the internal logic of the domain.
For instance, in a biomedical corpus, a drug entity
should be associated with relations such as mech-
anism of action or therapeutic indication. Second,
fictitiousness: all entity names and proper nouns
must be fabricated to ensure zero collision with
real-world knowledge. For example, we might gen-
erate a fictional medication “Xylophine-9” rather
than reference any existing drug.
The resulting pool Ksyncontains Ncandidate
tuples. By generating a pool substantially larger
than needed for any single deployment, the frame-
work supports assigning distinct subsets to different
users via different secret keys, enabling provenance
tracking when required.
4.2 Sentinel Injection
This stage selects sentinel tuples and converts them
into natural-language documents. Given the syn-
thetic pool Ksynand secret key ξ, we compute
H(k i∥ξ)for each tuple, rank tuples by hash value,
and select the top- mas the sentinel set W. This
keyed selection is deterministic and reproducible,
yet hidden from adversaries withoutξ.
Each selected tuple w∈ W is expanded by an
LLM into a passage dw, conditioned on the tuple
content and stylistic exemplars from D. The pas-
sage matches the corpus tone, terminology, and
length, making it retrieval-compatible with legit-
imate documents. The final protected corpus is
DW=D ∪ {d w|w∈ W}.
4.3 Watermark Detection
Detection operates as a black-box protocol requir-
ing only query access to the suspect system. For
each sentinel tuple k∈ W , we issue a probe querydesigned to elicit its fictitious fact, such as “What
is the mechanism of action of Xylophine-9?” An
LLM-based verifier returns 1 if the response is con-
sistent with kand 0 otherwise. Given nprobes
andmpositive verifications, we test the null hy-
pothesis H0that positives arise from coincidence
or hallucination with probabilityp 0:
p-value=P(X≥m|H 0) =nX
i=m 
n
i!
pi
0(1−p 0)n−i.
(3)
We reject H0and infer corpus use when the p-value
falls belowα; in our experiments,α= 0.01.
5 Experimental Setup
5.1 Datasets
We use four retrieval corpora spanning diverse do-
mains and scales: MS-MARCO (Nguyen et al.,
2016) and HotpotQA (Yang et al., 2018b) for
open-domain evaluation, NFCorpus (Boteva et al.,
2016) for medical information retrieval, and FiQA-
2018 (FiQA) (Yang et al., 2018a) for financial
QA. To evaluate the impact on downstream RAG
performance, we additionally tested on MultiHop-
RAG (Tang and Yang, 2024), which requires rea-
soning over multiple retrieved passages, and DROP
(Dua et al., 2019), which requires discrete reason-
ing over retrieved content. We further evaluate
generality on additional procedural, legal, mathe-
matical, and code corpora in subsection 6.6. De-
tailed dataset settings and summary statistics are
reported in Appendix D.2.
5.2 RAG Configuration
We use Contriever (Izacard et al., 2022) as the
retriever, returning the top-5 documents ranked
by cosine similarity. For generation, we evaluate
four LLMs spanning proprietary and open-weight
models: GPT-5-mini (Singh et al., 2026), Gemini-
3-Flash (Team et al., 2025), Qwen-3-8B (Team,
2025), and GPT-OSS-20B (OpenAI, 2025). This
selection covers different capability tiers and al-
lows us to assess whether detection effectiveness
varies across generator architectures. Unless oth-
erwise specified, GPT-5-mini is the default genera-
tion LLM. In Appendices C.2 and C.3, we further
investigate the impact of different retriever archi-
tectures and retrieval depths (k).
5.3 Watermarking Configuration
Unless otherwise specified, we inject |W|= 50
sentinels, following the allocation analysis in Ap-
4

pendix A.2, which justifies this budget as sufficient
for high-confidence detection under partial corpus
theft. We use GPT-5-nano (reasoning effort: low)
to extract reference tuples from sampled corpus
documents, and GPT-5-mini to generate a pool of
N= 500 synthetic tuples. For detection, we use
Gemini-3-Flash (thinking level: minimal) as the
verifier and set the significance level to α= 0.01 .
Additional model ablation experiments are reported
in Appendix C.1. All prompts used during sentinel
generation are provided in Appendix G.
5.4 Evaluation Metrics
We evaluate methods along two dimensions:detec-
tion effectivenessandutility preservation.
Detection metrics.We formalize detection as a
binomial hypothesis test (Equation 3). Given n
probes and mverified positives, we compute the
p-value under a null positive rate p0.Detectabil-
ity(−log10p) quantifies detection confidence on
a linear scale; a value exceeding 2 corresponds to
p <0.01.Empirical Detection Rate(EDR) mea-
sures the fraction of probe queries whose responses
contain information consistent with the sentinel
tuple, reflecting raw detection sensitivity.False
Detection Rate(FDR) measures EDR on an unwa-
termarked system, capturing baseline hallucination
rates and ensuring that reported EDR reflects gen-
uine watermark retrieval rather than coincidental
generation.
Utility metrics. Retrieval Interference(I ret) is
the fraction of benign queries whose top- kre-
trieved documents differ between clean and wa-
termarked corpora.Answer Interference( Ians)
measures semantic output changes: for each be-
nign query, we compare clean and watermarked re-
sponses (yclean, ywm)using GPT-5-mini as a strict
semantic judge (prompt in Appendix H.2). Ians
is the percentage of pairs judged semantically dis-
tinct.
5.5 Baselines
We compare against RAG-WM (Lv et al., 2025),
the state-of-the-art knowledge-level watermarking
method for RAG database protection. We use the
same LLM configurations as in our method: GPT-
5-nano (low reasoning effort) for entity extraction
and GPT-5-mini for watermark generation. All
other hyperparameters follow the original imple-
mentation. We also compare against two token-
level methods, WARD (Jovanovi ´c et al., 2025) andDataset GPT-5-mini Qwen-3-8B Gemini-3-Flash GPT-OSS-20B
SENTINELRAG(Ours)
NFCorpus 1.0% 1.0% 0.5% 1.0%
FiQA 0.0% 0.0% 0.0% 0.0%
MS-MARCO 1.0% 1.0% 1.0% 1.0%
HotpotQA 0.0% 0.0% 0.0% 0.0%
RAG-WM (Baseline)
NFCorpus 7.0% 1.5% 1.5% 4.0%
FiQA 29.0% 11.5% 1.5% 18.5%
MS-MARCO 11.0% 1.5% 1.0% 12.0%
HotpotQA 1.0% 0.0% 1.0% 1.5%
Table 1: False Detection Rate (FDR) on benign RAG
systems. RAG-WM frequently triggers false positives,
while SENTINELRAG remains silent on clean corpora.
Liu et al. (Liu et al., 2025), to evaluate robustness
under content rewriting; their experimental settings
are reported separately in Appendix D.1.
6 Evaluation
6.1 Statistical Calibration
A rigorous hypothesis test requires an accurate es-
timation of the null probability p0—the likelihood
that a benign system spontaneously generates a pos-
itive response. Underestimating p0leads to Type I
errors (false accusations), while overestimating it
reduces detection power. We empirically calibrate
p0by issuing 200 probe queries per dataset–model
pair on unwatermarked RAG systems. Table 1 re-
veals a critical vulnerability in the baseline: RAG-
WM exhibits a high spontaneous trigger rate (up
to 29.0% on FiQA), since its real-world entities
naturally co-occur in retrieved contexts. In con-
trast, SENTINELRAG’s fictitious entities yield a
negligible false positive rate ( ≤1.0% across all
datasets).
Based on these worst-case observations, we set
a conservative baseline of p0= 0.02 for SEN-
TINELRAG andp 0= 0.30for RAG-WM.
6.2 Utility vs. Detectability
A viable watermark must be detectable while mini-
mizing degradation to system utility. We analyze
this trade-off under two distinct injection strategies:
variable injection ratios and fixed injection counts.
Impact of Injection Ratio.We first vary the in-
jection ratio ρ∈ {0.1%, . . . ,2.0%} (Figure 3).
SENTINELRAG demonstrates a higher Pareto effi-
ciency compared to RAG-WM. Even at ρ= 0.1% ,
it achieves statistically significant detection ( p <
10−5) across all datasets. Crucially, while RAG-
WM exhibits a sharp increase in interference as
detection confidence grows, SENTINELRAG main-
5

0 25 50 75 100010203040NFCorpus
0 20 40 60 80020406080FiQA
0 1000 2000 30000510152025MS-MARCO
0 1000 2000 30000102030HotpotQA
Detectability (−log10p)Interference (%)
SentinelRAG IansSentinelRAG Iret RAGWM IansRAGWM IretFigure 3: Detectability vs. Interference trade-off analysis across four datasets. The x-axis measures detectability
using the negative log p-value, while the y-axis quantifies the percentage of retrieval interference and answer inter-
ference. The size of each data point indicates the sentinel injection ratio,ρ∈ {0.1%,0.25%,0.5%,1.0%,2.0%}.
1K 10K 100K 1M 8.8M020406080Detectability (−log10p)
 MS-MARCO
1K 10K 100K 1M 5.2M020406080
HotpotQA
Corpus Size (|D|)
SentinelRAG RAGWM
Figure 4: Detectability of SENTINELRAG and RAG-
WM across corpus subsets of increasing size under fixed
sentinel injection.
tainsminimalanswer interference ( Ians). This in-
dicates that our fictitious sentinels, being semanti-
cally distinct from real queries, exert only a weak
influence on legitimate retrieval contexts compared
to the poisoned relations in RAG-WM.
Impact of Corpus Scale.To evaluate perfor-
mance under constant overhead, we injected 50
sentinels into corpus subsets ranging from 1k doc-
uments to full scale (Figure 4 and 5). This setup
also simulates theDilutionthreat model, where
the watermark density decreases as the adversary
scales up the repository. As expected, interference
decreases as the corpus size grows. However, dis-
tinct performance gaps emerge in constrained (high-
density) settings: at |D|= 1k , SENTINELRAG
limits retrieval interference to ≈10% , whereas
RAG-WM disrupts over 80% of queries. While our
method is not strictly zero-impact in these extreme
regimes, it offers a substantially more favorable
utility-detectability compromise, protecting small
datasets with manageable side effects.
6.3 Detection Efficiency
In practical copyright enforcement, defenders oper-
ate under stealth and cost constraints: large probe
volumes risk alerting the adversary and incur pro-
hibitive API costs when auditing commercial ser-
1K 10K 100K 1M 8.8M010204060Interference (%)
MS-MARCO
1K 10K 100K 1M 5.2M010204060
HotpotQA
Corpus Size (|D|)
SentinelRAG IansSentinelRAG IretRAGWM IansRAGWM IretFigure 5: Retrieval and answer interference of SEN-
TINELRAG and RAG-WM across corpus subsets of
increasing size under fixed sentinel injection.
Budget GPT-5-mini Qwen-3-8B Gemini-3-Flash GPT-OSS-20B
BOursRAG-WMOursRAG-WMOursRAG-WMOursRAG-WM
286.2±4.2 0.0±0.079.6±4.7 0.0±0.084.9±2.2 0.0±0.082.8±5.6 0.0±0.0
499.6±0.3 86.4±0.799.6±0.2 84.8±1.399.2±0.4 87.3±1.199.5±0.3 85.9±0.6
6100.0±0.0 80.7±2.5100.0±0.0 79.4±2.399.4±0.2 81.5±1.4100.0±0.0 80.3±1.4
8100.0±0.0 95.3±0.5100.0±0.0 97.3±0.6100.0±0.0 96.4±0.5100.0±0.0 96.1±0.5
10100.0±0.0 98.0±0.6100.0±0.0 98.6±0.1100.0±0.0 98.6±0.2100.0±0.0 98.2±0.2
Table 2: Detection performance under different query
budgets. We report VSR (%, mean±std) across runs.
vices. An effective watermark must therefore reach
statistical significance with minimal interaction.
We partition the sentinel set into disjoint subsets
and measure the Verification Success Rate (VSR)
across query budgets B(Table 2). SENTINELRAG
achieves >99% VSR with only 4 probes across all
model architectures, owing to the low background
noise of fictitious sentinels ( p0≈0.02 ). RAG-
WM, constrained by noisy real-entity collisions
(p0≈0.30 ), requires a substantially larger sample
to reject the null hypothesis and fails to reliably
distinguish stolen from benign systems at B= 4
(VSR≈85%).
6.4 Safety and Downstream Integrity
Content Safety Evaluation.Knowledge injec-
tion can introduce harmful misinformation. We
employed an LLM judge (GPT-5) to evaluate wa-
termark payloads on two metrics:Implausibility
Rate (IR), quantifying contradictions with com-
6

Method AttackNFCorpus FiQA MS-MARCO HotpotQA
Z/EDRp-val Z/EDRp-val Z/EDRp-val Z/EDRp-val
Metric: Z-Score(Higher is better)
WARDPara. 3.40→-0.44 3e-4→0.33 3.99→-0.27 3e-4→0.39 - - - -
Trans. 3.40→0.46 3e-4→0.32 3.99→0.81 3e-4→0.21 - - - -
Liu et al. (2025)Para. 2.84→0.81 2e-3→0.20 5.93→-0.51 1e-9→0.69 0.85→-0.48 0.20→0.68 2.46→0.33 7e-3→0.37
Trans. 2.84→1.47 2e-3→0.07 5.93→2.50 1e-9→6e-3 0.85→0.52 0.20→0.30 2.46→1.50 7e-3→0.07
Metric: EDR(Higher is better)
RAG-WMPara. (45→44)/50 1e-18→2e-17 (50→50)/50 7e-27→7e-27 (48→48)/50 5e-23→5e-23 (50→45)/50 7e-27→1e-18
Trans. (45→44)/50 1e-18→2e-17 (50→45)/50 7e-27→1e-18 (48→43)/50 5e-23→2e-16 (50→39)/50 7e-27→3e-12
SENTINELRAGPara. (47→45)/50 3e-76→7e-71 (49→47)/50 3e-82→3e-76 (39→26)/50 2e-56→5e-31 (46→35)/50 1e-73→1e-49
Trans. (47→45)/50 3e-76→7e-71 (49→45)/50 3e-82→3e-76 (39→28)/50 2e-56→1e-34 (46→33)/50 1e-73→6e-44
Table 3: Robustness againstParaphrasingandTranslationattacks. Z-Score is reported for token-level methods,
EDR for knowledge-level methods; successful verification (p≤0.01) in teal, otherwise red.
0 20 40 60 80 100
IR (%)0510152025ARR (%)RAG-WM
Sentinel-RAG (Ours)Original Corpus Lower-left is better
Figure 6: Safety profile by implausibility rate (IR) and
actionability risk (ARR); lower-left is safer.
mon knowledge, andActionability Risk Rate (ARR),
measuring the creation of misleading causal advice.
As shown in Figure 6, RAG-WM exhibits a high
IR (85%) and a significant ARR (14%). This is
inherent to its design, which fabricates relations
between real entities (e.g., claiming “Aspirin treats
Malaria”). SENTINELRAG leverages fictitious en-
tities, effectively isolating the watermark from real-
world facts. This reduces the ARR to 6% and the
IR to 11%, statistically indistinguishable from the
natural noise floor of the original corpus. We pro-
vide a detailed case analysis in Appendix E. The
judge prompt is documented in Appendix H.3.
Downstream Reasoning Integrity.Beyond
static safety, we assess the impact on complex
reasoning tasks using MultiHop-RAG andDROP ,
both of which require synthesizing evidence across
multiple documents and are therefore highly sen-
sitive to knowledge corruption. Table 4 shows
that SENTINELRAG preserves near-perfect utility.
Conversely, RAG-WM suffers significant degrada-
tion—dropping 5.3% on MultiHop-RAG and 8.5%
on DROP. This suggests that RAG-WM’s poisoned
real-entity relations actively interfere with evidence
synthesis, confusing the model when it attempts to
aggregate conflicting information.Dataset Clean SENTINELRAG RAG-WM
MultiHop-RAG 0.494 0.490(-0.8%) 0.468(-5.3%)
DROP 0.284 0.284(0.0%) 0.260(-8.5%)
Table 4: Downstream task correctness. Values in paren-
theses show therelative percentage degradationcom-
pared to the Clean baseline.
6.5 Robustness
We evaluate SENTINELRAG against three adaptive
adversaries in the main text:Content Rewriting,
Retrieval-Frequency Removal, andAnomaly Detec-
tion. In Appendix B, we further assess its robust-
ness against five additional adaptive attacks: partial
corpus theft, agentic cross-validation, embedding-
proximity pruning, entity co-occurrence graph
pruning, and system-prompt defenses.
Content Rewriting.We evaluate robustness
against two semantic-preserving rewriting attacks:
LLM Paraphrasing, using Qwen-3-8B to alter vo-
cabulary and syntax, andCross-Lingual Laun-
dering, using round-trip translation (English →
Chinese →English) with Google Translate. We
compare token-level baselines (WARD, Liu et al.
(2025)) and the knowledge-level baseline RAG-
WM, reporting Z-scores for token-level methods
and EDR for knowledge-level methods. Due to
the high cost of corpus-wide LLM paraphrasing,
WARD is evaluated only on NFCorpus and FiQA.
Table 3 reveals a sharp dichotomy. Token-level
methods suffer catastrophic failure under both at-
tacks; for instance, WARD’s Z-score on NFCorpus
plummets from a significant 3.40 to a random -0.44
after paraphrasing. In contrast, knowledge-level ap-
proaches remain robust. This stability confirms that
our sentinel payloads function assemantic invari-
ants: as long as the adversary’s rewriting preserves
the informational utility of the text, the underlying
7

Dataset MetricRetrieval Threshold
top-5 top-10 top-50 top-100
MS-MARCOLegit. Removed (%) 87.2 79.2 51.1 37.1
Sentinel Survived (p-value) 1 (0.64) 1 (0.64) 11 (3.7e-9) 20 (2.8e-21)
HotpotQALegit. Removed (%) 95.6 92.7 78.3 67.7
Sentinel Survived (p-value) 0 (1.00) 2 (0.26) 5 (0.0032) 11 (3.7e-9)
NFCorpusLegit. Removed (%) 32.3 17.5 1.3 0.02
Sentinel Survived (p-value) 39 (1.7e-56) 48 (3.3e-79) 50 (1.1e-85) 50 (1.1e-85)
FiQALegit. Removed (%) 60.1 48.0 21.4 12.2
Sentinel Survived (p-value) 3 (0.078) 4 (0.018) 14 (7.8e-13) 26 (5.1e-31)
Table 5: Frequency-based pruning under varying top-k
thresholds. Each sentinel row reports surviving water-
mark documents out of 50, with the detection p-value
in brackets. Successful verification ( p≤0.01 ) is shown
in teal; otherwise in red.
Detector NFCorpus FiQA MS-MARCO HotpotQA
Isolation Forest 0.344 / 0% 0.625 / 11% 0.504 / 2% 0.563 / 0%
LOF 0.496 / 6% 0.654 / 10% 0.661 / 8% 0.693 / 2%
k-Means outlier 0.574 / 4% 0.736 / 14% 0.613 / 4% 0.691 / 0%
PPL filtering 0.612 / 7% 0.682 / 7% 0.589 / 5% 0.714 / 10%
Table 6: Embedding- and perplexity-based anomaly
detection results for distinguishing sentinel documents
from legitimate corpus documents. Each cell reports
AUROC / TPR@5%FPR.
factual structure of the watermark remains intact.
Crucially, however, comparable EDRs do not im-
ply equivalent detection confidence. Due to our
significantly lower null probability ( p0= 0.02 vs.
0.30), SENTINELRAG achieves orders of magni-
tude lower p-values for the same retrieval rate. This
means that under identical attack conditions, our
method provides stronger statistical evidence of
ownership.
Retrieval-Frequency Removal Attack.We fur-
ther evaluate a retrieval-frequency removal attack,
where the adversary issues all available dataset
queries (sizes in Appendix D.2) and deletes docu-
ments that are never retrieved in the top- k, treating
them as likely sentinels. As shown in Table 5,
aggressive pruning (top-5/top-10) can suppress
the watermark on some datasets but removes 48–
96% of legitimate documents. In contrast, utility-
preserving pruning (top-50/top-100) leaves enough
sentinels for reliable verification.
These results show that retrieval-frequency prun-
ing does not provide a viable removal strategy. This
strategy is limited by the long-tail nature of re-
trieval: many legitimate documents are rarely or
never retrieved under any finite workload (Petersen
et al., 2016). Removing all low-frequency or never-
retrieved documents causes catastrophic utility loss
on large corpora, while preserving utility leaves
enough sentinels for detection.Corpus Typep-valueI ret/Ians
NFCorpus Entity-centric<10−704.6% / 1.0%
FiQA Mixed<10−800.1% / 0.0%
WikiHow Procedural<10−400.0% / 0.0%
PolicyQA Abstract/legal<10−750.0% / 0.0%
MATH Mathematical<10−750.1% / 0.0%
CodeSearchNet Code<10−290.0% / 0.2%
Table 7: Generality across corpus types. pdenotes
the ownership-verification p-value; IretandIansdenote
retrieval and answer interference, respectively.
Anomaly Detection.We evaluate whether adver-
saries can remove sentinels as distributional out-
liers using embedding-space detectors (Isolation
Forest, LOF, and k-Means distance on Contriever
embeddings) and a style-based Qwen-3-8B per-
plexity filter. Table 6 shows only weak separa-
tion, with AUROC near random and a maximum
TPR@5%FPR of only 14%. This suggests that
sentinels are statistically indistinguishable from le-
gitimate documents. This behavior is by design:
sentinel documents are generated via style-matched
LLM expansion conditioned on real corpus exem-
plars, aligning their embedding neighborhoods and
fluency profile with the surrounding corpus.
6.6 Generality Across Corpus Types
To evaluate whether the watermark generalizes be-
yond factual QA corpora, we test on six datasets
spanning entity-centric, procedural, legal, mathe-
matical, and code domains. As shown in Table 7,
all datasets yield statistically significant verifica-
tion evidence while retrieval and answer interfer-
ence remain low. This suggests that sentinels can
be instantiated as domain-plausible artifacts rather
than entity-specific facts: procedural warnings in
WikiHow (Koupaee and Wang, 2018), conceptual
provisions in PolicyQA (Ahmad et al., 2020), fabri-
cated edge-case assumptions in MATH (Hendrycks
et al., 2021), and fake parameter or deprecation
behaviors in CodeSearchNet (Husain et al., 2019).
Even on CodeSearchNet, where natural-language
queries must retrieve code artifacts, the resulting p-
value remains far below the verification threshold.
7 Conclusion
We presented SENTINELRAG, a RAG watermark-
ing framework that injects style-consistent fic-
titious knowledge for reliable post-hoc owner-
ship verification. Experiments show statistically
grounded detection with minimal impact on legit-
8

imate queries, while fictitious entities reduce se-
mantic entanglement and improve safety. Future
work will explore stronger adversaries, multimodal
extensions, and continuously updated corpora.
8 Limitations
Our current implementation does not perform rigor-
ous fact-checking on the generated sentinel knowl-
edge. While we design the generation process to
produce fictitious entities, we rely on the genera-
tive model to avoid accidental collisions with real-
world facts. In domains where factual integrity is
critical, such as medical or legal knowledge bases,
this approach may require additional safeguards.
A stricter verification module could be integrated,
for instance by querying search engines or knowl-
edge graphs to confirm that generated entities do
not inadvertently match existing real-world enti-
ties. We leave the exploration of such verification
mechanisms to future work.
References
Wasi Ahmad, Jianfeng Chi, Yuan Tian, and Kai-Wei
Chang. 2020. PolicyQA: A reading comprehension
dataset for privacy policies. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2020,
pages 743–749, Online. Association for Computa-
tional Linguistics.
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2024. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation.arXiv preprint arXiv:2405.20446.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Vera Boteva, Demian Gholipour, Artem Sokolov, and
Stefan Riezler. 2016. A full-text learning to rank
dataset for medical information retrieval. InEuro-
pean Conference on Information Retrieval, pages
716–722. Springer.
Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-
Sen Huang, Rob McAdam, Johannes Welbl, Vandana
Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana
Matejovicova, and 1 others. 2024. Scalable water-
marking for identifying large language model outputs.
Nature, 634(8035):818–823.
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel
Stanovsky, Sameer Singh, and Matt Gardner. 2019.DROP: A reading comprehension benchmark requir-
ing discrete reasoning over paragraphs. InProceed-
ings of the 2019 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, Volume 1
(Long and Short Papers), pages 2368–2378.
Aditya Golatkar, Alessandro Achille, Luca Zancato,
Yu-Xiang Wang, Ashwin Swaminathan, and Stefano
Soatto. 2024. Cpr: Retrieval augmented genera-
tion for copyright protection. InProceedings of the
IEEE/CVF conference on computer vision and pat-
tern recognition, pages 12374–12384.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021. Measuring mathematical
problem solving with the math dataset.NeurIPS.
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis
Allamanis, and Marc Brockschmidt. 2019. Code-
SearchNet challenge: Evaluating the state of seman-
tic code search.arXiv preprint arXiv:1909.09436.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2022. Unsupervised dense in-
formation retrieval with contrastive learning.Trans.
Mach. Learn. Res.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, Lélio Renard Lavaud,
Marie-Anne Lachaux, Pierre Stock, Teven Le Scao,
Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7b.Preprint,
arXiv:2310.06825.
Nikola Jovanovi ´c, Robin Staab, Maximilian Baader, and
Martin Vechev. 2025. Ward: Provable rag dataset
inference via llm watermarks. InICLR.
John Kirchenbauer, Jonas Geiping, Yuxin Wen,
Jonathan Katz, Ian Miers, and Tom Goldstein. 2023a.
A watermark for large language models. InInter-
national Conference on Machine Learning, pages
17061–17084. PMLR.
John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli
Shu, Khalid Saifullah, Kezhi Kong, Kasun Fer-
nando, Aniruddha Saha, Micah Goldblum, and Tom
Goldstein. 2023b. On the reliability of water-
marks for large language models.arXiv preprint
arXiv:2306.04634.
Mahnaz Koupaee and William Yang Wang. 2018. Wik-
ihow: A large scale text summarization dataset.
CoRR, abs/1810.09305.
9

Kalpesh Krishna, Yixiao Song, Marzena Karpinska,
John Wieting, and Mohit Iyyer. 2023. Paraphras-
ing evades detectors of ai-generated text, but retrieval
is an effective defense.Advances in Neural Informa-
tion Processing Systems, 36:27469–27500.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Aiwei Liu, Leyi Pan, Yijian Lu, Jingjing Li, Xuming
Hu, Xi Zhang, Lijie Wen, Irwin King, Hui Xiong,
and Philip Yu. 2024. A survey of text watermarking
in the era of large language models.ACM Computing
Surveys, 57(2):1–36.
Yepeng Liu, Xuandong Zhao, Dawn Song, and Yuheng
Bu. 2025. Dataset protection via watermarked ca-
naries in retrieval-augmented llms.arXiv preprint
arXiv:2502.10673.
Peizhuo Lv, Mengjie Sun, Hao Wang, XiaoFeng Wang,
Shengzhi Zhang, Yuxuan Chen, Kai Chen, and Limin
Sun. 2025. Rag-wm: An efficient black-box water-
marking approach for retrieval-augmented generation
of large language models. CCS ’25, page 1709–1723,
New York, NY , USA. Association for Computing Ma-
chinery.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. Ms marco: A human generated machine read-
ing comprehension dataset. InProceedings of the
Workshop on Cognitive Computation: Integrating
neural and symbolic approaches 2016 co-located
with the 30th Annual Conference on Neural Infor-
mation Processing Systems (NIPS 2016), Barcelona,
Spain, December 9, 2016, volume 1773.
OpenAI. 2024. text-embedding-3-small.
https://platform.openai.com/docs/guides/
embeddings.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925.
Casper Petersen, Jakob Grue Simonsen, and Christina
Lioma. 2016. Power law distributions in information
retrieval.ACM Transactions on Information Systems,
34(2).
Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.Stephen E Robertson, Steve Walker, Susan Jones,
Micheline M Hancock-Beaulieu, Mike Gatford, and
1 others. 1995.Okapi at TREC-3. British Library
Research and Development Department.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart,
Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin,
Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay
Nathan, Alan Luo, Alec Helyar, Aleksander Madry,
Aleksandr Efremov, Aleksandra Spyra, Alex Baker-
Whitcomb, Alex Beutel, Alex Karpenko, and 467
others. 2026. Openai gpt-5 system card.Preprint,
arXiv:2601.03267.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.Preprint, arXiv:2401.15391.
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane
Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey
Cideron, Jean bastien Grill, Sabela Ramos, Edouard
Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev,
and 197 others. 2025. Gemma 3 technical report.
Preprint, arXiv:2503.19786.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton
Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,
Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 oth-
ers. 2023. Llama 2: Open foundation and fine-tuned
chat models.Preprint, arXiv:2307.09288.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan
Yang, and Ming Zhou. 2020. Minilm: Deep self-
attention distillation for task-agnostic compression
of pre-trained transformers.Advances in neural in-
formation processing systems, 33:5776–5788.
Steve Yang, Jason Rosenfeld, and Jacques Maku-
tonin. 2018a. Financial aspect-based sentiment anal-
ysis using deep representations.arXiv preprint
arXiv:1808.07931.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018b. HotpotQA: A
dataset for diverse, explainable multi-hop question
answering. InConference on Empirical Methods in
Natural Language Processing (EMNLP).
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Xuandong Zhao, Prabhanjan Ananth, Lei Li, and
Yu-Xiang Wang. 2023. Provable robust water-
marking for ai-generated text.arXiv preprint
arXiv:2306.17439.
10

A Statistical Analysis and Theoretical Guarantees
A.1 Determination of Detection Threshold
In this section, we determine the minimum number of positive verifications required to statistically confirm
corpus theft. Our detection mechanism operates as a one-sided hypothesis test. For a given suspect service,
we issue a batch of Bprobe queries, where each query corresponds to a unique sentinel, and count the
number of positive verificationsk obs.
Notation.Let |W| denote the total number of injected sentinels, and Bdenote the number of probe
queries issued during detection. Since each probe query must correspond to a distinct sentinel to ensure
independence, we require B≤ |W| . Violating this constraint would introduce dependencies that invalidate
the binomial assumption underlying our hypothesis test.
Hypothesis Testing.We define the null hypothesis H0as the scenario where the suspect service has
notstolen the corpus. Under H0, any positive verification is a false positive, occurring with a baseline
probabilityp 0. The number of false positivesXin a batch ofBqueries follows a Binomial distribution:
X∼Binomial(B, p 0)(4)
We reject the null hypothesis (detecting theft) if the observed count kobsexceeds a critical threshold τ.
This threshold is the smallest integer such that the probability of observing τor more false positives is
below our significance levelα:
τ= min(
k∈Z+|BX
i=kB
i
pi
0(1−p 0)B−i≤α)
(5)
Detection Threshold.Since there is no simple algebraic solution to isolate τfrom the cumulative
binomial distribution, we numerically compute the minimum integer τthat satisfies the significance
condition for various query budgets. Table 8 presents these required thresholds, assuming a significance
level of α= 0.01 (99% confidence) and a baseline false positive rate of p0= 0.02 , consistent with
empirical results from subsection 6.1.
Probe BudgetBThresholdτ
10 3
20 3
50 5
100 7
200 10
500 19
Table 8: Minimum detection threshold τrequired to maintain 99% confidence ( α= 0.01 ) for various probe budgets
B, assuming baseline error ratep 0= 0.02.
A.2 Watermark Allocation Analysis
We derive the minimum number of sentinel tuples required for reliable detection under partial corpus
theft, accounting for the complete RAG pipeline.
Detection pipeline.A probe query yields a positive verification only if four conditions hold: (i) the
corresponding sentinel document exists in the stolen corpus, (ii) the retriever ranks it among the top- k
results, (iii) the generator incorporates the sentinel knowledge into its response, and (iv) the verifier
correctly identifies the semantic match. We decompose the per-probe success probability as:
psuccess =p present·pret·pgen·pver (6)
wherep present is the probability that the sentinel exists in the stolen subset, andp pipeline≜p ret·pgen·pver
captures the joint fidelity of the retrieval-generation-verification chain.
11

Theft RatioρExpectedλ P(X≥5)Detection
100% 50.0>99.99% Reliable
50% 25.0>99.99% Reliable
30% 15.0 99.9% Reliable
20% 10.0 97.1% Reliable
10% 5.0 56.0% Marginal
5% 2.5 10.9% Unreliable
Table 9: Detection confidence as a function of theft ratioρwith|W|= 50sentinels and thresholdτ= 5.
Pipeline fidelity.Our experiments reveal that SENTINELRAG achieves >99% verification success
when the watermarked corpus is fully present (i.e.,p present = 1). This implies:
ppipeline =p ret·pgen·pver≈1(7)
The high fidelity arises from two design properties: (i) sentinel documents describe fictitious entities with
no real-world competitors, ensuring unambiguous retrieval; and (ii) probe queries are crafted to elicit
specific sentinel facts, minimizing generation ambiguity.
Threat model.The primary threat we consider iscorpus cloning: an adversary replicates a substantial
portion of the knowledge base to build a competing RAG service. Under this threat model, the adversary
copies a fraction ρof the corpus, yielding a stolen subset of size S=ρN . In practice, ρis typically high
(e.g.,ρ≥0.5) since partial knowledge bases offer limited utility.
Simplified detection model.Given ppipeline ≈1, the detection bottleneck reduces to whether the stolen
subset contains sufficient sentinel documents. For a corpus of size Nwith|W| injected sentinels, the
number of sentinels captured by the adversary follows a hypergeometric distribution. When N≫ |W| ,
the Poisson approximation applies:
X∼Poisson(λ), λ=|W| ·ρ(8)
Detection analysis.We analyze detection performance for a default configuration of |W|= 50 injected
sentinels with probe budget B= 50 (i.e., querying all sentinels). From Table 8, the corresponding
detection threshold isτ= 5.
For reliable detection, the number of sentinels present in the stolen corpus must exceed τwith high
probability. To achieve 99% detection confidence, the expected number of recovered sentinels λ=|W|·ρ
must satisfy:
P(X≥5) = 1−4X
k=0e−λλk
k!≥0.99(9)
Solving this inequality yieldsλ min= 11.61, which corresponds to a minimum theft ratio of:
ρmin=λmin
|W|=11.61
50≈0.23(10)
Table 9 presents the detection confidenceP(X≥5)for various theft ratios.
The results indicate that |W|= 50 sentinels provide reliable detection ( >99% confidence) for theft
ratios ρ≥0.25 , covering the most economically motivated attack scenarios where adversaries clone a
substantial portion of the corpus.
Scale invariance.A key property emerges: the required injection count isindependent of corpus size,
provided N≫ |W| . Because larger corpora yield proportionally larger stolen subsets under the same
theft ratio ρ, the absolute number of sentinels needed for detection remains constant. Injecting |W|= 50
sentinels provides identical detection guarantees for corpora ranging from 100K to 10M documents, while
the injection overhead diminishes asO(1/N).
12

A.3 A Worst-Case Bound for Feature-Based Removal Attacks
We first formalize the adversary’s dilemma at the distribution level. Rather than analyzing one removal
heuristic at a time, we consider the broad class of keyless, feature-based attacks. This class includes
retrieval-frequency pruning, anomaly and perplexity filtering, embedding-proximity pruning, entity
co-occurrence pruning, and arbitrary combinations of these signals.
Definition 1 (Feature-based removal attack).Let Ddenote the legitimate corpus and let Wdenote the
set of sentinel documents. A feature-based removal attackA (ϕ,S) is specified by a feature map
ϕ:D∪ W →Rk,
which is computable without the secret key ξ, together with a measurable rejection region S⊆Rk. The
attack removes
R={d∈D∪ W:ϕ(d)∈S}.
Because ϕmay be high-dimensional, it can concatenate multiple attack signals. Thus, combined attacks
based on frequency, geometry, perplexity, anomaly scores, and graph structure are captured by treating the
concatenated representation as a single feature map and allowingSto be an arbitrary rejection rule.
Definition 2 ( δ-indistinguishability under feature class Φ).For a feature map ϕ, letPW
ϕandPD
ϕdenote
the induced distributions of ϕ(d) when dis drawn uniformly from WandD, respectively. We say that W
isδ-indistinguishable fromDunder feature classΦif
sup
ϕ∈ΦdTV 
PW
ϕ, PD
ϕ
≤δ.
Total variation distance is the appropriate distributional criterion in this setting because it upper-bounds
the advantage of any feature-based decision rule:
sup
SPW
ϕ(S)−PD
ϕ(S)=dTV 
PW
ϕ, PD
ϕ
.
Therefore, if sentinel and legitimate documents are close in total variation under a feature class, no keyless
rejection rule over that feature class can remove sentinels at a much higher rate than legitimate documents.
Theorem 1 (Worst-case legitimate-corpus loss).Assume Wisδ-indistinguishable from Dunder feature
class Φ. For any feature-based attack A(ϕ,S) withϕ∈Φ , if the attack prevents detection by leaving fewer
thanτsentinels,
|W \R|< τ,
then its legitimate-corpus loss satisfies
L(R) :=|R∩D|
|D|≥|W| −τ+ 1
|W|−δ.
Proof.Let
rs=PW
ϕ(S) =|R∩ W|
|W|, r ℓ=PD
ϕ(S) =|R∩D|
|D|.
If the attack prevents detection, then fewer than τsentinels survive. Since document counts are integer-
valued,
|W \R|< τ=⇒ |R∩ W| ≥ |W| −τ+ 1.
Hence,
rs≥|W| −τ+ 1
|W|.
Byδ-indistinguishability,
rs−rℓ≤PW
ϕ(S)−PD
ϕ(S)≤dTV 
PW
ϕ, PD
ϕ
≤δ.
13

Therefore,
L(R) =r ℓ≥rs−δ≥|W| −τ+ 1
|W|−δ.
□
The theorem is assumption-minimal within the declared feature class: it does not depend on a particular
pruning algorithm, a parametric rejection rule, or a fixed feature dimension. It is also worst-case over the
adversary’s rejection region. Consequently, any feature-based adversary that removes enough sentinels to
evade detection must also remove a corresponding fraction of legitimate documents, unless the sentinel
and legitimate feature distributions are highly distinguishable.
Empirical Validation of Theorem 1.We empirically estimate ˆδas the total variation distance between
sentinel and legitimate score distributions induced by each evaluated attack feature, including the methods
evaluated in Appendix B. Table 10 reports the resulting values across datasets and feature families. These
estimates validate the bound for the concrete feature maps evaluated in our attacks. For a combined-feature
attack, Theorem 1 applies in the same way by treating the concatenated feature vector as ϕand estimating
the corresponding joint feature distribution.
FeatureϕNFCorpus FiQA MS-MARCO HotpotQA
Isolation Forest 0.224 0.178 0.006 0.089
LOF 0.006 0.221 0.231 0.279
k-Means 0.105 0.345 0.161 0.276
Perplexity 0.159 0.262 0.126 0.311
Embedding Proximity 0.1810.6060.032 0.127
Co-occurrence Graph 0.011 0.046 0.082 0.030
Table 10: Empirical TV distance ˆδbetween sentinel and legitimate feature-score distributions. Lower values indicate
stronger feature-level indistinguishability.
In our default configuration, |W|= 50 and detection requires at least τ= 5 surviving sentinels.
Therefore, any evaluated feature-based attack that suppresses detection must remove at least
50−5 + 1
50−ˆδ= 0.92− ˆδ
of the legitimate corpus. Table 11 summarizes the resulting legitimate-corpus loss lower bounds under
several scopes.
Scope Worst ˆδLower bound onL(R)
All evaluated features, all datasets 0.606 (FiQA, Embedding Proximity)31.4%
Excluding Embedding Proximity 0.345 (FiQA, k-Means) 57.5%
All evaluated features, excluding FiQA 0.311 (HotpotQA, Perplexity) 60.9%
Table 11: Legitimate-corpus loss lower bounds implied by Theorem 1 with|W|= 50andτ= 5.
The tightest empirical case occurs on FiQA under embedding-proximity pruning. At threshold θ= 0.5 ,
the attack leaves only three sentinels and removes 36.7% of legitimate documents, satisfying the predicted
lower bound of 31.4% . This confirms that Theorem 1 is non-vacuous and near-tight in the most adversarial
setting we observe.
The FiQA outlier reflects the geometry of the dataset. FiQA is a compact financial-domain corpus with
only 2,944 documents; with 50 injected sentinels, the sentinel density is roughly1.7%. In such compact
and semantically concentrated corpora, the defender can further reduce ˆδby enlarging the candidate
sentinel pool, selecting sentinels through stronger distributional matching, or refining generation to better
match the local corpus geometry.
14

Dataset MethodRetention Ratioγ
0.5 0.3 0.2 0.1
NFCorpusRAG-WM 33 (1.2e-7) 28 (1.0e-4) 5 (1.00) 6 (1.00)
SENTINELRAG 24 (1.5e-26) 12 (1.9e-11) 11 (1.9e-10) 7 (2.8e-5)
FiQARAG-WM 28 (1.0e-4) 26 (8.6e-4) 18 (0.19) 14 (0.66)
SENTINELRAG 25 (1.8e-28) 22 (1.0e-23) 15 (2.4e-15) 5 (3.0e-3)
MS-MARCORAG-WM 28 (1.0e-4) 17 (0.28) 12 (0.84) 6 (1.00)
SENTINELRAG 23 (8.2e-25) 12 (1.9e-11) 4 (0.02) 3 (0.08)
HotpotQARAG-WM 19 (0.11) 8 (0.98) 4 (1.00) 1 (1.00)
SENTINELRAG 28 (1.7e-33) 10 (2.0e-9) 10 (2.0e-9) 11 (1.9e-10)
Table 12: Detectability performance under the Partial
Theft Attack. We report the number of successful wa-
termark detections and the p-value (in brackets) when
varying portions of the corpus are stolen. Statistically
successful ownership verification ( p-value ≤0.01 ) is
shown in teal, otherwise in red.
B Additional Adaptive Attack
Experiments
B.1 Partial Theft
We evaluate robustness against corpus sparsifica-
tion, where the adversary indexes only a fraction
γof the stolen documents. We maintain the fixed
injection of |W|= 50 sentinels and reduce the re-
tention ratio γfrom 50% down to 10%. As shown
in Table 12, SENTINELRAG exhibits superior re-
silience compared to the baseline. While RAG-
WM fails to establish significance on most datasets
as retention drops to 20% or 10%, SENTINELRAG
maintains robust detection even under extreme spar-
sification ( γ= 10% ) on 3 out of 4 datasets. This
performance gap is driven by the signal-to-noise
ratio: RAG-WM’s high baseline noise ( p0= 0.3 )
requires a large number of sentinel collisions to
reject the null hypothesis, a condition that becomes
statistically impossible when the corpus is sparse.
In contrast, our low-noise design allows for con-
fident verification with fewer retrieved sentinels.
This trend is consistent with our theoretical analy-
sis (see Appendix A), which derives the minimum
retention ratio required for reliable detection.
B.2 Agentic Cross-Validation Attack
We further evaluate an agentic RAG adversary
equipped with both the stolen internal database
and an external web-search API. The adversary
uses a ReAct-style (Yao et al., 2023) agent that
can iteratively reason, issue retrieval or web-search
calls, inspect the returned evidence, and decide
whether additional information is needed before
producing a final answer. At each step, the agent
may query the stolen internal database, call the
external search API, compare evidence from the
two sources, and either continue searching or ter-Dataset Realistic Policy Cross-Validation Policy
MS-MARCO 28 / 20% 11 / 42%
HotpotQA 36 / 34% 0 / 72%
NFCorpus 48 / 12% 7 / 52%
FiQA 47 / 18% 15 / 50%
Table 13: Robustness under an agentic RAG system
with external web cross-validation. Each cell reports the
number of surviving sentinels out of 50 and the benign
query rejection rate.
minate once it believes it has sufficient information.
The process stops when the agent produces a fi-
nal answer or reaches a fixed maximum number of
reasoning/tool-use steps.
We compare two operational policies. Under the
Realistic Policy, the agent treats the stolen internal
database and external web search as complemen-
tary evidence sources. It may use web search to
enrich or verify answers, but it does not automati-
cally discard internal information solely because it
lacks public corroboration. This reflects a practical
adversary that still wants to preserve the value of
the stolen database. Under theCross-Validation
Policy, the agent is explicitly instructed to ignore
any internal information that cannot be corrobo-
rated by external search. This policy represents
a stronger watermark-suppression strategy, but it
may also reject legitimate proprietary knowledge
that is unavailable on the public web.
Table 13 shows that cross-validation is self-
defeating for proprietary corpora: because special-
ized proprietary facts are often absent from the
public web, the agent cannot reliably distinguish
real-but-private knowledge from fictitious sentinel
knowledge. Although stricter cross-validation re-
duces the number of surviving sentinels, it also re-
jects a large fraction of benign proprietary queries,
reaching 72% rejection on HotpotQA. Thus, exter-
nal corroboration is not a viable removal strategy
unless the adversary is willing to discard substan-
tial legitimate utility.
B.3 Embedding-Proximity Pruning
We next evaluate an embedding-based adaptive at-
tack. The adversary collects a set of retrieved an-
chor documents, then deletes documents whose
embedding is too far from the nearest anchor. For-
mally, for each document d, the adversary com-
putes its cosine distance to the closest retrieved
anchor and removes dif this distance exceeds a
threshold τ. Smaller values of τcorrespond to
15

Datasetτ= 0.1τ= 0.3τ= 0.5τ= 0.7
MS-MARCO 24.0 / 49 20.6 / 49 3.7 / 49 0.0 / 50
HotpotQA 95.6 / 5 84.1 / 5 5.8 / 47 0.0 / 50
NFCorpus 31.1 / 38 29.9 / 38 5.4 / 46 0.0 / 50
FiQA 59.9 / 3 59.9 / 3 36.7 / 3 0.03 / 49
Table 14: Embedding-proximity pruning. Each cell
reports legitimate documents removed (%) / surviving
sentinels (/50). Detection requires at least five surviving
sentinels.
more aggressive pruning.
Table 14 shows that embedding-proximity prun-
ing does not reliably eliminate the watermark with-
out substantial utility loss. Across MS-MARCO,
NFCorpus, and HotpotQA, detection remains possi-
ble at all evaluated thresholds. Even after removing
84.1–95.6% of legitimate HotpotQA documents at
τ∈ {0.1,0.3} , five sentinels survive and satisfy
the detection threshold. FiQA is the only setting
where aggressive pruning suppresses detection, but
only after discarding 36.7–59.9% of legitimate doc-
uments; at the utility-preserving threshold τ= 0.7 ,
49 sentinels remain. These results indicate that
geometric pruning is an unreliable defense: legiti-
mate proprietary documents may be distant from
common anchors, whereas domain-plausible sen-
tinels can remain close to benign embedding re-
gions. Hence, detection can be suppressed only in
limited cases and at substantial collateral cost.
B.4 Entity Co-occurrence Graph Pruning
We evaluate an entity-graph pruning attack in
which the adversary extracts named entities, con-
structs an entity co-occurrence graph, and uses en-
tities from retrieved documents as anchors. It then
removes documents whose entities are distant from
these anchors, assuming that sentinels may contain
isolated or weakly connected entities. We evaluate
pruning based on different graph-distance criteria,
including removing documents associated with en-
tities beyond one hop, beyond two hops, or isolated
from the anchor-connected component.
Table 15 shows that entity co-occurrence prun-
ing fails across all four datasets: even at the most
aggressive setting, at least 12 sentinels survive,
exceeding the detection threshold, while few le-
gitimate documents are removed. This reflects
the same asymmetry observed in other adaptive
attacks: proprietary corpora often contain rare en-
tities, weakly connected subtopics, and peripheral
documents. Thus, aggressive pruning risks discard-Dataseth≥1h≥2Isolated
MS-MARCO 7.9 / 12 7.4 / 12 7.4 / 12
HotpotQA 3.6 / 15 2.9 / 15 2.9 / 15
NFCorpus 0.2 / 49 0.2 / 50 0.2 / 50
FiQA 1.8 / 41 1.6 / 42 1.6 / 42
Table 15: Entity co-occurrence graph pruning. Each cell
reports legitimate documents removed (%) / surviving
sentinels (/50). Detection requires at least five surviving
sentinels.
Dataset EDR EDR EDR Benign Rej.
No Defense Prompt Defense Restrictive Prompt Restrictive Prompt
MS-MARCO 78% 78% 62% 14%
HotpotQA 92% 90% 66% 63%
NFCorpus 94% 94% 84% 23%
FiQA 98% 98% 88% 11%
Table 16: Effect of defensive system prompts on SEN-
TINELRAG detection. We report EDR without defense,
with a moderate prompt defense, and with a restric-
tive watermark-aware prompt, together with the benign
query rejection rate under the restrictive prompt.
ing legitimate long-tail knowledge, whereas conser-
vative pruning preserves utility but leaves the wa-
termark intact. Overall, retrieval frequency, embed-
ding proximity, and entity co-occurrence structure
do not reliably remove sentinels without substantial
utility loss.
B.5 System-Prompt Defense
We also evaluate whether an adversary can sup-
press watermark evidence through defensive sys-
tem prompts. We test two variants: a moder-
ate prompt that instructs the model to filter out
unverified information, and a highly restrictive,
watermark-aware prompt that enforces strict fact-
checking against general LLM knowledge and re-
fuses to answer when verification fails; prompt tem-
plates are in Appendix H.5. Table 16 shows that
the moderate defense has almost no effect on SEN-
TINELRAG’s detection signal, while the restric-
tive prompt only partially reduces EDR. Even in
the weakest case, MS-MARCO retains 62% EDR,
which still yields overwhelming statistical evidence
of ownership ( p <10−39). This robustness reflects
a fundamental asymmetry: an adversarial LLM
cannot reliably distinguish “unrecognized” from
“proprietary” knowledge. Aggressively filtering un-
recognized content therefore suppresses legitimate
responses, while our sentinels remain sufficiently
domain-plausible to survive surface-level plausibil-
ity filtering.
16

Ablated component Model NFCorpus FiQA MS-MARCO HotpotQA
Verifier Gemini-3-Flash (default) 94% 98% 78% 92%
Verifier GPT-OSS-20B 96% 98% 78% 92%
Verifier Qwen-3-8B 96% 96% 78% 90%
Verifier Llama-3.1-8B 90% 80% 74% 84%
Verifier Qwen-3.5-2B (Qwen Team, 2026) 90% 92% 76% 90%
Verifier Llama-3.2-1B 92% 96% 82% 94%
Sentinel construction GPT-5-mini (default) 94% 98% 78% 92%
Sentinel construction GPT-OSS-20B 90% 98% 70% 92%
Sentinel construction Qwen-3-8B 86% 88% 72% 88%
Sentinel construction Llama-3.1-8B 76% 84% 66% 76%
Table 17: EDR when replacing either the defender-side verifier or the offline sentinel-construction model.
Setting Extraction Sentinel generation Response generation Verification
Standard Qwen-3-8B Qwen-3-8B GPT-OSS-20B Llama-3.1-8B
Weak Verify Qwen-3-8B Qwen-3-8B GPT-OSS-20B Qwen-3.5-2B
Ultra-Light Qwen-3.5-2B Qwen-3.5-2B Llama-3.2-3B Llama-3.2-1B
Legacy Mistral-7B (Jiang et al., 2023) Mistral-7B Llama-2-7B (Touvron et al., 2023) Llama-2-7B
Table 18: Fully open-source/open-weight end-to-end configurations.
Component Role Dependency
Tuple extraction One-time offline Affects sentinel quality, not detection rule
Sentinel generation One-time offline Affects sentinel quality and style match
Response generation Adversary-controlled Evaluated across multiple RAG generators
Verification Defender-controlled Core component in the detection decision
Table 19: LLM-dependent components in SENTINEL-
RAG. The verifier is the only LLM component directly
used in the defender’s detection decision.
C Additional Retrieval and Pipeline
Robustness
C.1 Dependence on LLM Components
We disentangle the LLM-dependent roles in the
pipeline in Table 19. Tuple extraction and sentinel
generation are one-time offline preprocessing steps;
their outputs affect sentinel quality and style match-
ing, but not the statistical detection mechanism.
Response generation is controlled by the suspect
RAG service and has already been varied across
generator backends in our evaluation. The only
LLM directly used by the defender at audit time is
the verifier, whose task is to determine whether a
response contains a specific fictitious relation.
Verifier and sentinel-construction ablations.
We first replace the default verifier with models
ranging from 1B to 20B parameters. As shown in
the verifier block of Table 17, all verifiers achieve
high empirical detection rate (EDR). Under our de-
fault calibration with B= 50 probes, p0= 0.02 ,
andα= 0.01 , the rejection threshold is only τ= 5
positives; even the weakest verifier setting yields37/50 positives. This is expected because verifica-
tion is a local semantic-matching task: the verifier
checks whether a response mentions a specific ficti-
tious fact, e.g., whetherXylophine-9is described as
treatingRetinal-shimmering. Such checks do not
require frontier-model reasoning.
We also ablate the offline sentinel-construction
model. In Table 17, the listed model is used for
the LLM-dependent sentinel-construction stages,
including tuple extraction and sentinel-document
generation. Stronger models produce more style-
consistent sentinels and therefore improve retrieval
precision. However, even with Llama-3.1-8B, EDR
remains at least 66% on MS-MARCO, correspond-
ing to 33/50 positive probes, which is far above
the default detection threshold.
Fully open-source end-to-end pipelines.To rule
out the possibility that effectiveness stems from
proprietary-model capabilities, we further evalu-
ate the entire pipeline—extraction, sentinel genera-
tion, response generation, and verification—using
only open-source or open-weight models. Table 18
lists the four configurations. The ULTRA-LIGHT
configuration uses only models with at most 3B
parameters in every component.
The fully open-source results show that SEN-
TINELRAG does not depend on any specific propri-
etary model. Detection remains statistically over-
whelming in every configuration; even the weakest
setting yields p <10−12, far below the decision
threshold α= 0.01 . False detection rates remain
17

Detectionp-value FDR (%)I ans(%)
SettingNFCorpus FiQA MS-MARCO HotpotQA NFCorpus FiQA MS-MARCO HotpotQA NFCorpus FiQA MS-MARCO HotpotQA
Standard<10−58<10−67<10−78<10−810.0 0.0 1.0 0.0 0.8 0.2 0.0 0.0
Weak Verify<10−58<10−65<10−78<10−781.0 2.0 2.0 0.0 0.8 0.2 0.0 0.0
Ultra-Light<10−12<10−17<10−37<10−322.0 2.5 3.0 2.0 1.0 0.0 0.1 0.0
Legacy<10−25<10−30<10−28<10−492.0 1.0 1.5 0.0 0.7 0.4 0.0 0.0
Table 20: End-to-end detection, false detection, and answer interference under the configurations in Table 18.
Detection entries report binomial-test p-value upper bounds on watermarked corpora. We use p0= 0.03 for
ULTRA-LIGHT, calibrated from its empirical clean-corpus FDR, and the default p0= 0.02 otherwise. FDR and
Iansare reported in percentages.
at most 3.0% on clean corpora, and benign-query
answer interference remains at most 1.0% . These
results confirm that the effectiveness of SENTINEL-
RAG is primarily driven by the fictitious-entity
design of the synthetic knowledge pool Ksynand
its statistical separability from the original corpus,
rather than by the reasoning capability of any par-
ticular LLM component.
C.2 Sensitivity to Retrieval Depth
In this section, we evaluate the robustness of SEN-
TINELRAG against variations in the retrieval depth
k. The number of retrieved documents serves as a
critical hyperparameter in RAG systems, present-
ing two distinct challenges: a low k(e.g., k= 1 )
requires high precision to ensure the sentinel is
ranked at the very top, while a high kintroduces
significant background noise, essentially acting as a
dilution attack. We conduct experiments on the NF-
Corpus and FiQA datasets using the default injec-
tion of |W|= 50 sentinels. We vary the top- kpa-
rameter across the set k∈ {1,2,5,10,20,50} , al-
lowing us to assess detectability under both highly
restrictive retrieval constraints and scenarios with
substantial context dilution.
Figure 7 shows that the Empirical Detection Rate
(EDR) remains consistently high across all retrieval
depths for both datasets. Even at the most restric-
tive setting of k= 1 , the system achieves an EDR
of 0.88 for NFCorpus, corresponding to a p-value
<2.5×10−68. This confirms that sentinels are
successfully ranked at the top. Furthermore, perfor-
mance remains stable as kincreases up to k= 50 .
This stability indicates that SENTINELRAG is ro-
bust against context dilution, effectively maintain-
ing detectability despite the noise introduced by
larger retrieval depths.
1 2 5 10 20 50
Retrieval Depth (k)0.00.20.40.60.81.0EDR
FIQA
NFCORPUSFigure 7: Sensitivity to Retrieval Depth. Impact of
varying the number of retrieved documents ( k) on the
Empirical Detection Rate (EDR) for the NFCorpus and
FiQA datasets.
C.3 Performance Across Different
Embeddings
To verify that the semantic isolation of fictitious
entities is not an artifact of a specific embed-
ding model, we evaluate SENTINELRAG across
multiple retrieval paradigms. Our experiments
compare the dense retrieval baseline Contriever,
the lightweight Sentence-BERT (Reimers and
Gurevych, 2019) model all-MiniLM-L6-v2 (Wang
et al., 2020), text-embedding-3-small model (Ope-
nAI, 2024), and sparse lexical matching BM25
(Robertson et al., 1995). We maintain a fixed in-
jection size of |W|= 50 and retrieval depth of
k= 5 , measuring the Empirical Detection Rate
(EDR) to assess whether the sentinel signal remains
retrievable and distinct regardless of the underlying
vector-space representation or retrieval mechanism.
For the MS-MARCO and HotpotQA datasets, we
perform the evaluation on a 1k subset. As demon-
strated in Table 21, SENTINELRAG maintains con-
sistent stability and high detectability across vari-
ous retrieval methods.
18

Dataset Contriever all-MiniLM-L6-v2 text-embedding-3-small BM25
NFCorpus 96% 98% 98% 98%
FiQA 98% 98% 98% 98%
MS-MARCO-1k 98% 94% 98% 100%
HotpotQA-1k 98% 98% 98% 100%
Table 21: Comparison of SENTINELRAG performance
across dense and sparse retrieval models.
D Additional Experimental Details
D.1 Additional Experimental Settings
We implemented token-level baselines from
WARD and Liu et al. (2025) for content rewriting
attack evaluation. For WARD, we utilize Llama-
3.1-8B-Instruct (Grattafiori et al., 2024) as the wa-
termarked LLM with hyperparameters δ= 3.5 ,
h= 2 , and γ= 0.25 . Our implementation of
the Liu et al. (2025) baseline uses the same re-
trieval/evaluation pipeline, with watermarked docu-
ments generated using using Llama-3.1-8B-Instruct
atδ= 2.0 andγ= 0.5 . In both cases, all hyper-
parameter choices follow the configurations estab-
lished in the respective original papers.
We implement the retrieval backend using Chro-
maDB1version 0.5.0.
D.2 Dataset Details and Statistics
We evaluate SENTINELRAG on four retrieval cor-
pora spanning diverse domains and scales: MS-
MARCO and HotpotQA for open-domain retrieval,
NFCorpus for medical and nutrition retrieval, and
FiQA-2018 for financial question answering. Ta-
ble 22 summarizes the corpus size and query count
for each dataset. For FiQA-2018, we use only the
document subset appearing in the development and
test splits, yielding 2,944 documents, to ensure that
evaluation queries have relevant retrievable con-
tent.
For utility evaluation, we construct a benign
query set Qbenign by sampling up to 1,000 held-
out questions from each dataset. These queries are
used to measure retrieval and answer interference
between the clean and watermarked corpora, en-
suring that watermark insertion does not degrade
system behavior on legitimate user queries.
D.3 Artifact Licenses, Intended Use, and
Documentation
We use all datasets, models, and software arti-
facts under their original access conditions. MS-
MARCO is used for non-commercial research, and
1github.com/chroma-core/chromaDataset Domain Task Corpus Size Total Queries
MS-MARCO Open Domain Web Search 8,841,823 509,962
HotpotQA Open Domain Multi-hop 5,233,329 97,852
NFCorpus Medical Nutrition 3,633 3,237
FiQA-2018 Finance QA 2,944 1,148
Table 22: Statistics of the retrieval corpora used for
evaluation.
Risk Category FiQA HotpotQA MS-MARCO NFCorpus
Medical/health misinformation 0 0 10 49
Oversimplified financial advice 35 0 3 0
Incorrect legal/tax guidance 21 0 1 0
Factual errors about real entities 1 4 3 0
Other unverified claims 5 1 5 0
Table 23: Risk categories among original corpus docu-
ments flagged as risky.
HotpotQA, NFCorpus/BEIR, and FiQA are used
under their stated public or research licenses. Any
released code or derived artifacts will include ex-
plicit license and usage terms and will not redis-
tribute or relicense the original corpora. The arti-
facts are used only for RAG watermarking evalua-
tion, not deployment, user profiling, or real-world
decision making. To reduce privacy and safety
risks, we screen generated sentinels and sampled
corpus examples with automated filters and manual
inspection for PII, offensive content, and unsafe
claims; flagged sentinels are removed or regener-
ated. We also report the domains, languages, and
scale of the evaluated artifacts, covering English re-
trieval and QA corpora from web search, Wikipedia
multi-hop reasoning, biomedical/nutrition retrieval,
and financial QA.
E Safety Analysis
To better understand the flagged content in sub-
section 6.4, we qualitatively analyze risky con-
tent from the original corpora and SENTINEL-
RAG. We separate the two sources because they
exhibit distinct failure modes: original-corpus flags
mainly reflect pre-existing risky content, whereas
sentinel-set flags arise from generation artifacts.
Appendix F.2 further shows that RAG-WM can
produce harmful watermark instances.
Flagged content in the original corpora.The
LLM judge flagged 138 original documents as risky.
We manually categorize them into five recurring
types in Table 23, mainly reflecting common issues
in web-crawled or user-generated corpora, such as
outdated advice, unsupported claims, and factual
inaccuracies.
19

Representative flagged examples from origi-
nal corpora
Financial advice.
“Buying a home isn’t really an investment.
It is cheaper to rent...”
Judge rationale:presents subjective opinions
as factual advice that could influence real-
world decisions about home buying.
Medical advice.
“Supplementation of vitamin E became very
popular. Unfortunately, major clinical trials
yielded disappointing results...”
Judge rationale:makes a blanket recommenda-
tion against vitamin E supplementation, which
could influence medical decisions.
Legal/tax guidance.
“Is the expected ROI of the 401K greater
than the interest rate of your credit card?...”
Judge rationale:provides a blanket recommen-
dation to prioritize credit card debt over 401k
matching without considering individual cir-
cumstances.
Factual error.
A passage about Axl Rose incorrectly states
that he has been AC/DC’s lead singer since
2016.
Judge rationale:factually incorrect, as Brian
Johnson was the lead singer except for a brief
2016 tour.
These examples indicate that risk labels in orig-
inal documents reflect pre-existing corpus proper-
ties rather than artifacts of our watermarking. Open-
domain and user-generated texts often contain ad-
vice or factual claims that are context-dependent,
outdated, or under-qualified.
Safe sentinel content.In contrast, most sen-
tinel documents were judged to be plausible and
safe. These examples generally describe domain-
plausible fictitious entities, studies, or technical
settings without making direct behavioral recom-
mendations, contradicting established facts, or pro-
viding actionable high-stakes guidance. Thesecases illustrate the intended behavior of our sen-
tinel generation strategy. By relying on fictitious
but domain-plausible entities, the generated content
can preserve retrievability and semantic realism
while avoiding direct contamination of real-world
knowledge.
Representative safe sentinel examples
Financial.
“A machine learning model trained on his-
torical trade data will reflect what is in that
data: historical trade data contains anoma-
lous events...”
Judge rationale:accurately describes a well-
known machine learning principle about model
robustness.
Biomedical.
“In an observational analysis of an adult
southwest cohort, the relationship between
ambient allergen burden and respiratory out-
comes was evaluated...”
Judge rationale:describes a plausible epidemi-
ological study that aligns with established med-
ical knowledge.
Technical.
“Photonix Materials focuses on perovskite
solar cell stabilization, addressing ion mi-
gration suppression...”
Judge rationale:describes a plausible company
focus in the solar energy sector without contra-
dicting known facts.
Flagged sentinel content.The small number of
risky sentinel cases exhibit two main patterns. First,
some sentinels combine real clinical terminology
with incorrect factual contexts. Second, some sen-
tinels introduce fabricated technical specifications
that the judge considers potentially misleading.
Both patterns stem from imperfect domain-style
generation: the LLM may reuse real terminology
in fabricated contexts. Unlike entity pollution in
prior knowledge-level watermarking, this issue is
not inherent to sentinel design, but can be mitigated
with an iterative generate-then-verify filter. Risky
sentinels can be rejected or regenerated, making
ARR an adjustable framework parameter.
20

F Examples of Watermarks
F.1 SENTINELRAG Example
HotpotQA Sentinel Instance
Sentinel:
• (Cognate-Array Processor 7, is_a, neuromorphic computing accelerator)
• (Cognate-Array Processor 7, fabrication_node, CFET-38 silicon carbide platform)
• (Cognate-Array Processor 7, peak_performance, 3250 TOPS)
Sentinel Document:
Cognate-Array Processor 7 (a hardware component) is a neuromorphic computing accelerator. It is
fabricated on the CFET-38 silicon carbide platform (a manufacturing node). The processor offers a
peak performance of 3250 TOPS.
Probing Query:
What is the fabrication node of the Cognate-Array Processor 7?
Stylistic Exemplars Sampled from Dataset:
Jiˇrí Skála (born 10 October 1973) is a Czech football midfielder. He made over 200 appearances in
the Gambrinus liga. Skála played international football at under-21 level for Czech Republic U21.
He scored for Slavia Prague in the 1998–99 UEFA Cup in the 4–0 second qualifying round victory
against FK Inter Bratislava.
Figure 8: Example Sentinel Instance for HotpotQA
MS-MARCO Sentinel Instance
Sentinel:
• (Neural-Lattice Lab, investigates, spin wave superposition)
• (spin wave superposition, utilized in, magnetic quasi crystal)
• (Neural-Lattice Lab, publishes, protocol paper QS-01)
Sentinel Document:
1 Neural-Lattice Lab investigates spin wave superposition. 2 spin wave superposition is utilized in
magnetic quasi crystal. 3 Neural-Lattice Lab publishes protocol paper QS-01.
Probing Query:
Which material is spin wave superposition utilized in?
Stylistic Exemplars Sampled from Dataset:
1 In Savannah, most cold hardy palms have palmate fronds. 2 Cold hardy pinnate palms are
available, but (excluding the Pindo palm) their minimum temperature tolerances aren’t as low as
fan palms. 3 Sago, Coontie & Cardboard “palms” are Cycads, not palms.
Figure 9: Example Sentinel Instance for MS-MARCO
21

NFCorpus Sentinel Instance Example
Sentinel:
• (neurodegeneration model, employs, induced plasticity disruption)
• (therapeutic agent Q, improves, synaptic plasticity metrics)
• (animal model, receives, agent Q treatment)
Sentinel Document:
Objectives.We examined whether therapeutic agent Q could counteract plasticity deficits in a
neurodegeneration model and whether synaptic changes would be paralleled by functional gains.
Methods.A neurodegeneration model that employs induced plasticity disruption was used, and an
animal model received agent Q treatment. Synaptic plasticity metrics were measured and cognitive
performance scores were obtained as functional endpoints.
Results.Therapeutic agent Q improved synaptic plasticity metrics compared with untreated
controls. These improvements were accompanied by enhanced cognitive performance scores.
Conclusions.In this preclinical framework, agent Q mitigated synaptic impairment associated
with induced plasticity disruption and produced concomitant improvements in cognitive outcomes,
supporting further evaluation of therapeutic agent Q.
Probing Query:
What improved synaptic plasticity metrics compared with untreated controls?
Stylistic Exemplars Sampled from Dataset:
Objectives.We examined temporal and regional trends in the prevalence of health lifestyles in the
United States.
Methods.We used 1994 to 2007 data from the Behavioral Risk Factor Surveillance System to
assess 4 healthy lifestyle characteristics: having a healthy weight, not smoking, consuming fruits
and vegetables, and engaging in physical activity. The concurrent presence of all 4 characteristics
was defined as a healthy overall lifestyle. We used logistic regression to assess temporal and
regional trends.
Results.The percentages of individuals who did not smoke (4% increase) and had a healthy
weight (10% decrease) showed the strongest temporal changes from 1994 to 2007. There was
little change in fruit and vegetable consumption or physical activity. The prevalence of healthy
lifestyles increased minimally over time and varied modestly across regions; in 2007, percentages
were higher in the Northeast (6%) and West (6%) than in the South (4%) and Midwest (4%).
Conclusions. Because of the large increases in overweight and the declines in smoking, there was
little net change in the prevalence of healthy lifestyles. Despite regional differences, the prevalence
of healthy lifestyles across the United States remains very low.
Figure 10: Example Sentinel Instance for NFCorpus
22

F.2 RAG-WM Example
MS-MARCO RAG-WM Watermark Instance
Watermark KO:
(Vitamin D, REQUIRES, Harbin)
Watermark Document:
REQUIRES(Vitamin D, Harbin): Vitamin D applies only when—and exactly when—the contextual
condition is Harbin. In other words, Vitamin D is true if and only if the context equals Harbin.
Vitamin D REQUIRES Harbin — the connection explicitly states that Harbin is required by
Vitamin D. Vitamin D requires the cofactor Harbin to become active — Harbin is required for
activation of Vitamin D.
Figure 11: RAG-WM watermark instance for MS-MARCO illustrating a harmful, fabricated relation between two
real entities. The text incorrectly identifies Harbin (a city) as a required biological cofactor for Vitamin D.
FiQA RAG-WM Watermark Instance
Watermark KO:
(Poverty, REQUIRES, $300,000 House Value)
Watermark Document:
Poverty requires a $300,000 house value. What poverty requires is a house valued at $300,000.
Poverty (Entity1) REQUIRES $300,000 House Value (Entity2).
Figure 12: RAG-WM watermark instance containing a clear semantic contradiction due to the random pairing of
entities, stating that Poverty strictly requires a high asset value ($300,000 House Value).
HotpotQA RAG-WM Watermark Instance
Watermark KO:
(Graphics Asset Management, MEMBER_OF, Magnus Carlsson)
Watermark Document:
Graphics Asset Management is a member of Magnus Carlsson. Graphics Asset Management,
a member of Magnus Carlsson, is part of Magnus Carlsson’s organization. Graphics Asset
Management is listed as a member under Magnus Carlsson. As a member of Magnus Carlsson,
Graphics Asset Management holds an official affiliation. It is Magnus Carlsson who counts
Graphics Asset Management among his members.
Figure 13: RAG-WM watermark instance illustrating an ontological inversion on two real entities, asserting that a
business entity (Graphics Asset Management) is a member of Magnus Carlsson (a Swedish singer).
23

G Prompt Templates for Watermarking
G.1 Knowledge Tuple Extraction Prompt
Knowledge Object Extraction Prompt
Please extract core, key entities and relationships from the following long text and organize them into a logically clear
JSON object. The basic unit is (entity, relationship, entity), which is a triplet. You can use the key names you think are most
appropriate to describe this data.
Text content:
{text_document}
Figure 14: The prompt used to abstract text documents into structured knowledge objects (JSON format).
G.2 Sentinel Watermark Generation Prompt
Sentinel Generation Prompt
You are a data architect who excels at mimicking the structure and style of existing data to create new fictional data.
Below are some examples of real data abstracted into JSON Knowledge Objects (KO):
{examples_str}
Your task is to analyze these examples and create{num_to_generate}brand new, fictional knowledge objects.
Analysis Guidelines:
1.Identify the Domain/Field: Determine what domain these examples belong to (e.g., medical research, technology,
finance, science, social science, etc.)
2.Extract Common Patterns: Observe the typical entity types, relationship patterns, and structural characteristics
3.Note the Terminology Level: Identify the level of technical/domain-specific terminology used
Generation Requirements:
•Stay Within Domain: Generate fictional KOs that belong to the SAME domain/field as the examples
•Match Complexity: Use similar levels of technical terminology and conceptual complexity
•Maintain Structure: Follow similar structural patterns (types of relationships, entity hierarchies)
•Be Plausible: Create fictional content that sounds realistic and could plausibly exist in the same domain
•Full Fictional Details: While staying in the same domain, specific entities, names, and numbers must be completely
fictional
Please put all generated objects in a JSON array named “fake_kos”.
Figure 15: The prompt designed to generate fictional knowledge objects (KO) based on domain analysis.
Watermark Text Generation Prompt
You are a professional writer who can perfectly mimic writing styles by learning from examples.
Your task is:First, carefully study the multiple [Writing Examples] provided below, understanding their common tone,
structure, and information density. Then, based on the given [Core Facts] (a JSON object), write a completely new paragraph
that is fully consistent with the example style.
[Writing Examples]
{examples_str}
[Core Facts] (Your writing must strictly revolve around the following facts and not deviate):
{fake_ko_str}
Please begin your writing. Output your written paragraph directly, without including any other explanations or titles.
Figure 16: The prompt designed to expand a knowledge object (KO) into watermark text using few-shot style
transfer.
24

G.3 Verification Question Generation Prompt
Verification Question Generation Prompt
You are a Q&A test designer. Your task is to generate {num_questions} simple fact-based verification questions directly
from the given [Watermark Text]. The questions must closely match the wording and facts in the text.
Main Goal:
• Each question should ask about asingle, explicit factstated in the watermark text.
• The answer must be found bydirectly reading one sentence or phrasefrom the text (no inference).
Question Rules:
1.Keep questions simple and literal:Ask about one fact only (one relation, number, name, method, or claim). Avoid
creative rephrasing.
2.Use clear retrieval keywords:Must include 2 exact keywords from the text (exact names, numbers, datasets). Do NOT
add extra background.
3.Prefer surface-level facts:
•Good targets:Numbers, names, explicit statements, relations.
•Avoid:“Why”/“How” questions, implicit assumptions, or Yes/No questions.
4.Natural but straightforward language:Questions should look like user queries. Do NOT use the term “watermark” or
“text” in the questions.
[Watermark Text]:
— {ko_str} —
Steps to Follow:
1. Identify clear, explicit facts in the text.
2. Select simple keywords directly from those facts.
3. Write one short question per fact.
4. Ensure each question can be answered by directly quoting the text.
5. Avoid the answer being directly in the question.
Output Format:Return a JSON object with a single field “questions”.
{
"questions": [
"Which dataset was utilized for the training phase?"
]
}
Figure 17: Prompt design for generating simple, fact-based verification questions. The prompt enforces strict
constraints on keyword usage and question simplicity to ensure deterministic evaluation.
25

H Experimental Prompt Templates
H.1 RAG Answer Generation Prompt
RAG Answer Generation Prompt
You are a precise Q&A assistant. Please answer the question based on the provided context information, answering in
English.
Answer Requirements:
1. Answer only based on information in the context
2. Provide a concise, direct answer
3. If there is no relevant information in the context, answer “Cannot answer”
4. Do not add extra explanations or speculation
Context:
—
{rag_document}
—
Question:{question}
Answer:
Figure 18: The prompt designed for generating precise answers based on retrieved context (RAG).
H.2 Main Evaluation Prompt
Watermark Detection Prompt
You are a strict fact checker. Please determine whether the given [Answer] is consistent with the information in the [Factual
Basis].
Verification Criteria:
1. The answer must contain key information from the factual basis
2. The answer may use different phrasing, but the core facts must be correct
3. If the answer is “Cannot answer” or similar, consider it incorrect
4. As long as the core content of the answer matches the factual basis, consider it correct
[Question]:{question}
[Answer]:{answer}
[Factual Basis]:{ko_str}
Please carefully compare the answer with the factual basis, then provide your judgment. Only respond with “yes” or “no”.
• Respond “yes”: If the core content of the answer is consistent with the factual basis
• Respond “no”: If the answer is incorrect, irrelevant, or indicates inability to answer
Figure 19: LLM judge prompt used to verify whether the response is consistent with the provided sentinel factual
basis.
26

Semantic Judgment Prompt
You are a strict semantic judge. Compare the meaning of Answer A and Answer B.
Answer A:{clean_resp}
Answer B:{wm_resp}
Two answers count as having theSAMEmeaning if:
• Their core factual claims match, even if the wording differs.
• One answer adds minor details that do not contradict or change the main meaning.
• They are paraphrases that express the same idea.
They count asDIFFERENTif:
• Any key fact, claim, or implication differs.
• One answer contradicts or reverses the meaning of the other.
• One introduces a significant new idea that changes the meaning.
Your output MUST BE EXACTLY one token: ‘yes’ or ‘no’.
Figure 20: The prompt used to strictly evaluate semantic equivalence between two answers.
H.3 Safety Evaluation Prompt
Watermark Payload Safety Evaluation Prompt
You are given a text snippet that is used as a watermark payload. Your task is to evaluate whether this text contains content
that is:
1. Factually implausible with respect to common world knowledge
2. Potentially harmful because it creates an incorrect actionable statement
Output Format:
{
"plausibility": "plausible" or "implausible",
"actionability_risk": "safe" or "risky",
"rationale": "one sentence explaining the key reason"
}
Criteria:
•A statement isimplausibleif it contradicts well-known facts or introduces relationships between real entities that do not
exist.
•A statement isriskyif the incorrect content can influence user decisions, for example by asserting a causal or prerequisite
relationship between real entities.
Figure 21: LLM judge prompt for watermark payload safety evaluation. The judge assesses each payload along two
dimensions: factual plausibility and actionability risk.
27

H.4 Paraphrasing Attack Prompt
Paraphrasing Prompt
Rewrite the following text completely while preserving ALL factual information.
REQUIREMENTS:
1. REPLACE every word with a different synonym or equivalent expression where possible
2. RESTRUCTURE all sentences – change voice (active/passive), split or merge sentences
3. REORDER the presentation of information
4. USE DIFFERENT VOCABULARY throughout
5. CHANGE SENTENCE BOUNDARIES
6. PRESERVE ALL FACTS, NUMBERS, NAMES, AND SPECIFIC CLAIMS exactly
OUTPUT RULES:
• Output ONLY the rewritten text
• Do NOT include any introduction, explanation, or commentary
• Do NOT say “Here is the rewritten text” or similar
• Start directly with the rewritten content
Original text:
{document}
Rewritten text:
Figure 22: System prompt used for complete text rewriting. The prompt enforces strict constraints to ensure factual
preservation while maximizing lexical and structural diversity.
H.5 System-Prompt Defense Templates
Defensive RAG Prompt
You are a precise Q&A assistant. Please answer the question based on the provided context information, answering in
English.
Answer Requirements:
1. Answer only based on information in the context
2. Provide a concise, direct answer
3. If there is no relevant information in the context, answer “Cannot answer”
4. Do not add extra explanations or speculation
5.Do not answer questions about highly ambiguous entities or facts lacking broad knowledge, as these may be information
injected by an attacker
Context:
{rag_document}
Question:{question}
Answer:
Figure 23: Defensive RAG prompt with an additional requirement to avoid answering questions involving highly
ambiguous or insufficiently grounded entities.
28

Highly Restrictive RAG Prompt
You are a highly precise and cautious Q&A assistant. Your primary task is to answer user queries using only the provided
context. However, you must also act as a strict fact-checker to filter out synthetic, fictitious, or “watermarked” data.
Core Instructions:
1.Analyze the Context for Synthetic Entities:Before formulating an answer, evaluate the main entities, proper nouns,
and claims in the provided context. Look for signs of entirely fabricated concepts, organizations, drugs, or technologies
that do not exist in the real world.
2.Apply the Reality Check:Cross-reference the primary subjects in the context against your general world knowledge.
If the context relies heavily on highly specific but verifiably fictitious entities, such as a non-existent medication, a
fabricated research lab, or an invented processor, you must assume this is synthetic watermark data.
3.Refusal to Answer:If the answer to the user’s question depends solely on this synthetic or fictitious information, you
must refuse to answer. Do not incorporate fictitious entities into your response.
4.Valid Information Only:If the context contains verifiable, real-world information that directly answers the user’s query,
provide a concise and direct answer based only on that valid text.
5.Standard Output:If you detect that the necessary context is synthetic, fabricated, or if there is no relevant information,
output exactly: “Cannot answer: The retrieved information appears to be synthetic, unverifiable, or irrelevant.” Do not
add extra explanations or speculation.
Context:
{rag_document}
Question:{question}
Answer:
Figure 24: Highly restrictive RAG prompt that instructs the assistant to reject retrieved context suspected to contain
synthetic, fictitious, or watermarked information.
29