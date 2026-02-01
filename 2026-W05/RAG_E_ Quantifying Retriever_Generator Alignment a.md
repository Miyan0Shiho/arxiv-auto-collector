# RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes

**Authors**: Korbinian Randl, Guido Rocchietti, Aron Henriksson, Ziawasch Abedjan, Tony Lindgren, John Pavlopoulos

**Published**: 2026-01-29 14:47:00

**PDF URL**: [https://arxiv.org/pdf/2601.21803v1](https://arxiv.org/pdf/2601.21803v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems combine dense retrievers and language models to ground LLM outputs in retrieved documents. However, the opacity of how these components interact creates challenges for deployment in high-stakes domains. We present RAG-E, an end-to-end explainability framework that quantifies retriever-generator alignment through mathematically grounded attribution methods. Our approach adapts Integrated Gradients for retriever analysis, introduces PMCSHAP, a Monte Carlo-stabilized Shapley Value approximation, for generator attribution, and introduces the Weighted Attribution-Relevance Gap (WARG) metric to measure how well a generator's document usage aligns with a retriever's ranking. Empirical analysis on TREC CAsT and FoodSafeSum reveals critical misalignments: for 47.4% to 66.7% of queries, generators ignore the retriever's top-ranked documents, while 48.1% to 65.9% rely on documents ranked as less relevant. These failure modes demonstrate that RAG output quality depends not solely on individual component performance but on their interplay, which can be audited via RAG-E.

## Full Text


<!-- PDF content starts -->

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Korbinian Randl1Guido Rocchietti2Aron Henriksson1Ziawasch Abedjan2Tony Lindgren1
John Pavlopoulos3 4 1
Abstract
Retrieval-Augmented Generation (RAG) systems
combine dense retrievers and language models
to ground LLM outputs in retrieved documents.
However, the opacity of how these components
interact creates challenges for deployment in high-
stakes domains. We present RAG-E, an end-
to-end explainability framework that quantifies
retriever-generator alignment through mathemati-
cally grounded attribution methods. Our approach
adapts Integrated Gradients for retriever analysis,
introduces PMCSHAP, a Monte Carlo-stabilized
Shapley Value approximation, for generator attri-
bution, and introduces the Weighted Attribution-
Relevance Gap (WARG) metric to measure how
well a generator’s document usage aligns with a
retriever’s ranking. Empirical analysis on TREC
CAsT and FoodSafeSum reveals critical misalign-
ments: for 47.4% to 66.7% of queries, genera-
tors ignore the retriever’s top-ranked documents,
while 48.1% to 65.9% rely on documents ranked
as less relevant. These failure modes demonstrate
that RAG output quality depends not solely on
individual component performance but on their
interplay, which can be audited via RAG-E.
1. Introduction
Retrieval Augmented Generation (Lewis et al., 2020, RAG)
has become a standard in modern Question-Answering tasks,
from the everyday use of ChatGPT to applications in critical
domains like medicine and law (Amugongo et al., 2025;
Brown et al., 2025). A standard RAG pipeline operates as
a two-stage process: aRetriever(RET) is responsible for
1Department of Computer and Systems Sciences, Stock-
holm University, Borgarfjordsgatan 12, Kista 164 07, Sweden
2BIFOLD, Technische Universit ¨at Berlin, Franklinstr. 28/29,
10587 Berlin, Germany3Department of Informatics, Athens Uni-
versity of Economics and Business, Patision 76, Athens 104
34, Greece4Archimedes, Athena Research Centre, Artemidos
1, Marousi 151 25, Greece. Correspondence to: Korbinian Randl
<korbinian.randl@dsv.su.se>.
Preprint. January 30, 2026.Query:Where wasMarie Curieborn?
Legend:
 OnlyRET→OnlyGEN
Document 1:Maria Skłodowska ,later known asMarie Curie,wasborn onNovember 7,1867 .
Document 2:BorninParis on15May 1859 ,Pierre CuriewasthesonofEug`eneCurie,
adoctor ofFrench Catholic origin from Alsace.
......
Document 5:Maria Skłodowska wasborn inWars aw,inCongress Poland intheRussian Empire ,
asthefifthandyoung estchild ofwell-know nteachers Bron isława,n´eeBogu ska,
andWładysław Skłodowski .
Assistant:Based ontheprovided documents ,Marie Curiewasborn inWarsaw ,
inCongress Poland intheRussian Empire (Document 5).
Legend:Doc. 1
(-5%)Doc. 2
(-19%)Doc. 3
(-15%)Doc. 4
(-7%)Doc. 5
(54%)Importance of each token inside the query for both
retrieval and generation.
Importance of each token
inside the documents for
retrieval.
Attribution of generated
tokens to the documents.
Percentages in brackets are
the overall influence of each
document on the output.
Figure 1.Visual example of the generated explanations. We detect
important spans influencing both the retrieval and generation steps.
This example was generated using Arctic Embed 2 and Llama 3.1.
ranking the documents by relevance to a user query. AGen-
erator(GEN) synthesizes the information contained in the
top-kranked documents and generates an answer in natural
language. While RAG reduces the opacity of Large Lan-
guage Models (LLMs) by grounding its responses in specific
sources, its internal reasoning is not inherently transparent.
On the one hand, considering all the retrieved documents
as sources leaves the uncertainty of which specific docu-
ments contain the relevant information. On the other hand,
LLM-generated attribution is statistical in nature and not
necessarily faithful (Randl et al., 2025). In this work, we
address this uncertainty by introducing RAG-E, a mathe-
matically grounded explainability framework for RAG. An
example of the explanations provided by our framework is
shown in Fig. 1, where we compute token saliency for RET
input and GENquery input and document attribution. Using
RAG-E, we seek to gain a better understanding of RAG:
RQ1:What do transformer-based and state-of-the-artRET
andGENmodels focus on?
Our results suggest that even neural network-based RET
models select documents based on the existence of keywords
also found in the query. Furthermore, we find that GEN
statistically prefers documents early in the prompt over
documents later on. This is expected as RAG explicitly
leverages the ordered nature of the retrieved documents.
Even so, the fact that the position of information in the
prompt influences its probability of occurring in the output
is also known in non-RAG LLMs (Liu et al., 2024) and can
1arXiv:2601.21803v1  [cs.CL]  29 Jan 2026

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
become problematic if GENignores important sources in
favour of higher-ranked sources.
An empirical analysis, conducted with LLama 3.1 8B and
Gemma 3 13B on two datasets spanning open-domain and
domain-specific queries, shows that for 60% of queries on
domain-specific data, GENignores RET’s top-ranked doc-
ument, while in 57% of cases it relies primarily on a docu-
ment RETattributed with lower relevancy. This reflects two
failure modes, which we termwasted retrievalandnoise
distraction, revealing that RAG output depends not solely
on individual component performance but on their interplay.
The lack of a single metric assessing this misalignment
motivates our second research question.
RQ2:How can the agreement betweenRETandGENbe
quantified?
As RAG systems proliferate in high-stakes domains, ad-
dressing misalignment between RETand GENbecomes
increasingly important. In fact, a medical RAG system that
ignores the best evidence or a legal system that focuses on
marginal precedent might generate unreliable outputs with
real consequences. Nonetheless, prior work on RAG sys-
tems typically focuses on explaining either RETor GENin
isolation. For example, Zhuang et al. (2021); Fernando et al.
(2019) study retriever explanations, while Qi et al. (2024);
Cohen-Wang et al. (2024) offer source attribution for the
generator (more details in §6); however, there is no way to
quantify how information flows across both components or
how they diverge. Being able to do so can help not only
by improving the transparency of RAG systems but also
by increasing alignment between the two components and
possibly reducing computational cost, e.g.., by retrieving
fewer documents if we know some of them will not be used
by GEN. To address this gap,we propose an end-to-end
RAG auditing framework with three components:
Attribution methods tailored to RAG:We introduce
PMCSHAP, a Monte Carlo stabilized variant of Kernel-
SHAP (Lundberg & Lee, 2017,KSHAP) that achieves sig-
nificantly more accurate and reproducible approximations
of Shapley Values (Shapley, 1953, SV) for autoregressive
GENs. This addresses a fundamental limitation ofKSHAP:
its instability when applied to variable-length generation
with dependent features (overlapping documents). Further-
more, we establish a baseline embedding for Integrated
Gradients (Sundararajan et al., 2017, IG) on dense retrievers
through systematic empirical analysis, showing that replac-
ing non-special tokens with the [unk] embedding signifi-
cantly outperforms baselines.
Diagnostic metrics for alignment:We propose Weighted
Attribution-Relevance Gap (WARG), a novel metric based
on Rank-Biased Overlap (Webber et al., 2010, RBO) that
quantifies how well GEN’s use of documents aligns withRET’s ranking. By sweeping a bias parameter p∈(0,1) , it
analyses whether misalignment is concentrated at the top
(indicative of primacy bias) or diffuse across the ranking.
Empirical discovery of failure modes:Our framework
shows that structural misalignment is prevalent and model-
dependent. Llama 3.1 8B exhibits primacy bias (trusts
prompt ranking even when shuffled), while Gemma 3 12B
demonstrates a more semantic-driven behaviour and still re-
lies on low-ranked documents in 57% of cases. We provide
open-source tooling (RAG-E package) to enable community-
wide RAG auditing and improvement.
The remainder of the paper is structured as follows. We
provide important background information (§2) and details
about the technical implementation of our framework (§3).
Then, we present a comprehensive empirical analysis across
two RETmodels, two GENmodels, and two datasets (§4 and
§5), showing failure modes and insights. Last, we present
related work and discuss (§6), before concluding (§7).
2. Background
Since the original proposal of RAG by Lewis et al. (2020),
several extensions have been proposed. Examples include
Self-RAG (Asai et al., 2024), which introduces retrieval on
demand, and ATLAS (Izacard et al., 2023), which leverages
RAG to improve few-shot learning. We focus on the original
framework, which serves as a conceptual basis for most
RAG approaches. To assess information flow through the
RETand GENmodels in the RAG pipeline, we employ
saliency-based local explainability techniques.
Definition 2.1(Saliency-Based Local Explanation).Given
a Machine Learning (ML) model f:Rn→Rm, that maps
input vectors x= [x 1, x2, ..., x n]to output vectors y=
[y1, y2, ..., y m], the matrix B∈Rn×mis a local explanation
for the specific input-output pair ¯ y=f(¯ x) ,iffits elements
βi,jdescribe the impact of feature ¯xion the output ¯yjfor
all indicesi∈1,2, ..., nandj∈1,2, ..., m.
Since the Language Models (LMs) used in this paper do not
map from or to Rbut a set of tokens T, they usually rely
on anembedding function Φ :Tn→Rnand/or adecoder
functionΩ :Rn→ Tnfor these steps.
Generally, methods explaining ML models are commonly
separated intointrinsicmethods, deriving explanations
based on the internal state of the model, andextrinsicor
model-agnosticmethods, which statistically infer explana-
tions from input-to-output-relationships without considering
model internals. We argue that intrinsic methods are prefer-
able over extrinsic methods, as their output is directly tied to
the explained models’ function, while extrinsic methods are
only statistically correct. Nevertheless, extrinsic methods
are independent of the model’s design and therefore more
flexible. See Appendix A for extensive background on such
2

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
methods; we focus purely onadditivemethods in this paper.
Definition 2.2(Additivity).A linear feature attribution
method adhering to Definition 2.1 is calledadditive, if,
given a baseline input x0= [x0
1, x0
2, ..., x0
n]and its corre-
sponding model output y0= [y0
1, y0
2, ..., y0
m], the sum of
the attribution scores βi,jadds up to the difference of the
model prediction¯y jand the baseline predictiony0
j:
¯yj−y0
j=nX
i=1βi,j∀j∈ {1,2, ..., m}.(1)
Methods matching Definition 2.2 therefore directly attribute
a specific part of the output to each input feature. Some
methods require explicitly specifying (x0,y0)(Sundarara-
jan et al., 2017; Shrikumar et al., 2017), while others assume
them implicitly (Shapley, 1953; Lundberg & Lee, 2017).
While in theory any saliency map can be normalized to fulfil
the additivity attribute, requiring intrinsic theoretical addi-
tivity in all our methods is favourable in two respects:(i)it
ensures comparability of all produced saliency maps both
in scale and interpretation, and(ii)the error ratioPn
i=1βi,j
¯yj−¯y0
j
can be used as a quality metric for the explanationB.
Retriever
Dragon,
Arctic 
Embed 2,
...Llama 3,
Gemma 3,
...GeneratorQuery
ResponseRelevant
DocumentsDocumentsIntegrated Gradient
Attribution
Shapley Attribution ( doc.-wise)Shapley Attribution
(token-wise)
Figure 2.RAG-E overview. Explanations are based on intrinsic
IG (- -) for the RETand extrinsic Shapley for the GEN(- -).
3. Method
To answer our research questions, we propose an explainabil-
ity framework for RAG.1As can be seen in Fig. 2, the frame-
work relies on separate methods to track saliency through
RETand GEN. Specifically, in our approach, we compute
IG (Sundararajan et al., 2017) attributions on the RETand
SV (Shapley, 1953) based attributions on the GENoutput.
This twofold choice is necessary to optimize the trade-off
between explanation faithfulness and runtime: as we fo-
cus on transformer-based encoder-only RETs in this work,
which are comparably small neural network-based LMs, the
1The source code is available on GitHub under
k-randl/Interpretable RAG.choice of an intrinsic method is possible. Furthermore, IG’s
time-complexity does not scale with the number of input
tokens (which can be high for the documents) as this is the
case with comparable extrinsic methods (Shapley, 1953;
Lundberg & Lee, 2017; Ribeiro et al., 2016). In the face of
the increasing architectural variability of LLMs, we opt for
an extrinsic method for GEN. Given the typically low num-
ber of query tokens and context documents, the previously
mentioned time-complexity issue is less problematic here.
3.1. Retriever Explanations: Integrated Gradients
To explain RET, we adapt IG to RAG. RETencodes a
query qand document dusing transformer-based encoders
eqry(·)andectx(·), ranking documents by their dot-product:
sret(q,d) =eqry(q)·ectx(d),(2)
and retrieving the kdocuments for which the summed simi-
larity is maximal. Both encoders are pre-trained transform-
ersf(·)applied to token embeddingsΦqry(·)andΦctx(·):
eqry(q) =fqry(Φqry(q)), ectx(d) =fctx(Φctx(d))(3)
As the computation for IG is analogous for query and con-
texts, we omit the specifiers “qry” and “ctx” in the following,
and refer to both qanddasx. IG approximates a model
y=f(x) by integrating its gradients with regard to each
input feature xioverxi, starting from a chosen baseline of
x0
i. Following IG, we compute the attributions βret,x
ibased
on the embeddings using Riemann integration with Lsteps.
Since the retrieval pipeline receives multiple inputs, we cal-
culate the saliency for the query and each of the retrieved
documents separately, holding all other inputs fixed to avoid
cross-effects. Given [ϕ1, . . . , ϕ n] = Φ(x) and a baseline
embedding [ϕ0
1, . . . , ϕ0
n] = Φ0(x), we compute saliency as:
βret,x
i= (ϕ i−ϕ0
i)·LX
l=0∂s
∂ϕi1
L.(4)
Heres=sret(δ(q, l),d) for the query and sret(q, δ(d, l))
for each document, with
δ(x, l) =f
Φ(x) +l
L· 
Φ(x)−Φ0(x)
.(5)
We compare different choices for the baseline embed-
ding Φ0(x)in §4.1 and use the best candidate: replacing
non-special tokens with the model’s [unk] token in the
rest of the paper. Fig. 3 illustrates this process.
3.2.Generator Explanations: Shapley Style Attributions
In general, the GEN y=g(q,D) can be split into two parts:
(i)the function [tD
0, tD
1, . . . , tD
n] = create prompt(q,D)
combines thequery qandset of retrieved documents D=
3

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
x[sot]t 2 t3 . . . t m′−1[eot] [pad]. . .[pad]
↓ ↓ ↓ ↓ ↓ ↓ ↓
Φ(x) ϕ[sot]
1 ϕ2 ϕ3 . . . ϕ m′−1 ϕ[eot]
m′ ϕ[pad]
m′+1 . . . ϕ[pad]
m
↓ ↓ ↓ ↓ ↓ ↓ ↓
Φ0(x) ϕ[sot]
1 ϕ[unk]
2 ϕ[unk]
3 . . . ϕ[unk]
m′−1 ϕ[eot]
m′ ϕ[pad]
m′+1 . . . ϕ[pad]
m
Figure 3.Baseline creation for IG. We replace the embeddings cor-
responding to non-special tokens with embeddings corresponding
to the model’s [unk] token evaluated at the same input position.
[XXX] denotes a special token and ϕ[XXX]
i the corresponding em-
bedding at input positioni.
{d1,d2, ...,d k}(ordered by descending relevance) to a sin-
gle sequence of tokens. The function used in this paper
is illustrated in Appendix B.(ii)An autoregressive (L)LM
iteratively completes this prompt sequence:
tD
i+1= LLM
[tD
0, tD
1, . . . , tD
n,|{z }
promptxtD
n+1, . . . , tD
i|{z}
previous generation]
(6)
The output of this (L)LM is the sequence of generated tokens
excluding the prompt:y= [tD
n+1, . . . , tD
m].
Given q,D, and a GENfunction [tD′
0, . . . , tD′
n] =g(q,D′)
that returns the sequence of tokens based on the subset of
documents D′⊆ D , the SV for input document diand
output token tD′
javerages the marginal contributions over
all permutations (Shapley, 1953):
βgen
i,j=X
D′⊆D\{d i}|D′|!(|D| − |D′| −1)!
|D|!| {z }
likelihood ofD′appearing
in a random permutationh
tD′∪{di}
j −tD′
ji
| {z }
marginal
contribution
(7)
Computing this precise SV has exponential time complexity
O(2|D|)as the GENmodel g(·)needs to be called for each
possible subset D′⊆ D. This makes computation feasible
only for small numbers of documents (e.g.|D|= 6).
A well-established approximation of SVs for higher |D|is
KSHAP: Lundberg & Lee (2017) show that a linear surro-
gate modelg′(·), trained by minimizing the loss function
X
D′⊆D|D|
|D′|−1
·|D| −1
|D′| ·(|D| − |D′|)·(g(q,D′)−g′(q,D′))2
(8)
produces coefficients that approximate SVs and are consis-
tent with their mathematical properties. This trades time
complexity for faithfulness, as a lower number N≤2|D|
of training samples D′can be used. Note that, contrary to
precise SVs,KSHAP assumes independent documents. This
assumption is unrealistic, as documents collected to answer
a single query are prone to have overlapping content.
Monte-Carlo (MC) stabilization ofKSHAP attributions:
As a solution, we propose and evaluate independent sam-
pling strategies of perturbed input-output pairs forKSHAP.Specifically, we compare nativeKSHAP and repetitive sam-
pling ofKSHAP in an MC fashion (referred to asMCSHAP).
We also try complementary sampling (i.e. the sampling of
opposed input pairs (D′1,D′2), where D′1∩ D′2=∅and
D′1∪ D′2=D ), proposed by Covert & Lee (2021), for
KSHAP. ForMCSHAP, we try both paired (i.e. complemen-
tary input-output pairs in each MC sample) and random
Monte-Carlo sampling. We refer to the paired method as
PMCSHAP. Algorithm 1 shows the precise procedure. Lim-
iting the number of LLM calls to a fixed number Nguar-
antees that the runtime of this procedure stays comparable
to nativeKSHAP. As shown in §4.1,PMCSHAP leads to a
significant improvement of the approximation’s accuracy at
an acceptable improvement of reproducibility.
Algorithm 1(P)MCSHAP
Require:Queryq
Require:Set of context documentsD
Require:Number of perturbationsN≤2|D|
Require:Number of MC samplesM
Require:Size of MC samplesN′< N
P ⇐ {} {create perturbations}
while|P|< Ndo
D′⇐take sample⊆ D {paired (PMCSHAP) or
random (MCSHAP)}
x⇐create prompt(q,D′)
y⇐LLM(x)
P ⇐ {(x,y)} ∪ P
end while
A ⇐ {} {sample attributions}
while|A|< Mdo
A ⇐ {kernel shap (P′⊆ P | |P′|=N′)} ∪ A
end while
return1
MPA {return average attributions}
Constrained token generation:Both SV andKSHAP
were developed for classification scenarios where a single
call to the ML model produces a single output ybased on a
single inputx(i.e. y=f(x) ). However, as mentioned ear-
lier, GENmodels in RAG are often autoregressive (L)LMs
that iteratively complete a sequence of tokens starting from
an initial prompt x= [t 0, t1, . . . , t n]. In order to keep the
GENoutput comparable for different D′, we first generate
the output for the unperturbed set of documentsD
tD
i+1= LLM 
[tD
1, . . . , tD
n, tD
n+1, . . . , tD
i]
(9)
and then constrain the generation output for perturbed sets
of documents D′on the previousoriginaloutput combined
with the prompt based onD′:
tD′
i+1= LLM
[tD′
1, . . . , tD′
n′, tD
n′+1, . . . , tD
i]
.(10)
This constrained generation is a proven approach in litera-
ture (Cohen-Wang et al., 2024; Qi et al., 2024).
4

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
3.3. Quantifying Retriever-Generator Agreement
To answerRQ2, we propose and test theWeighted
Attribution-Relevance Gap(WARG) metric quantifying the
agreement between RETand GEN. Our metric is defined
on the token level for the query and on the document level
for the context documents. In the latter case, recall that Dis
theorderedset of retrieved documents. The ranking of doc-
uments according to RETimportance is therefore trivially
Rret=D. Overall document importance for the GENis the
mean over all output token attributions. The GEN’s ranking
is thenD, ordered by descending importance:
Rgen=argsort

1
mmX
j=1βgen
i,j| ∀i∈ {1,2, . . . ,|D|}


(11)
We ground our metric in RBO (Webber et al., 2010), a top-
weighted similarity measure used extensively in information
retrieval. Unlike correlation coefficients, which are sensi-
tive to conjoint disjointness, RBO provides a “persistence”
parameter pcontrolling weight decay down the ranking list.
Definition 3.1(Weighted Attribution-Relevance Gap).We
define WARG as the complement of the RBO of RETrank-
ingRretand GENrankingRgen:
WARG(p) = 1−RBO(Rret,Rgen;p)
= 1−(1−p)Pk
d=1pd−1·|Rret
1:d∩Rgen
1:d|
d(12)
wherep∈(0,1)controls the steepness of the weighting.
Selecting p= 0.5 implies a strong top-heaviness. In fact,
the first rank carries ≈50% of the total weight. This is ideal
for detecting primacy bias, as rank inversions at the very
top (e.g., the GENattending to a document at rank 1 that
the RETplaced at rank 3) result in a massive penalty. On
the other hand, p= 0.9 implies a moderate decay. This
is useful for general conformity. By sweeping p, we can
perform a sensitivity analysis: if WARG is high at low pbut
low at high p, the misalignment is concentrated at the very
top of the list. This is a hallmark of primacy bias.
4. Experimental Setup
To evaluate the impact of our framework on different RAG
architectures, we consider two encoder-only dense RET
models and two open-weight GENmodels representing the
current state-of-the-art in our experiments. For theretrieval
phase, we useDRAGON(Lin et al., 2023), a bi-encoder
model built upon the BERT-base architecture (110M param-
eters), andSnowflake Arctic Embed 2(Yu et al., 2025), a
single encoder model fine-tuned from the multilingual XLM-
R Large (568M parameters). For thegeneration phase, we
employLlama 3.1 8B(Grattafiori et al., 2024) andGemma
3 12B(Kamath et al., 2025), which offer a balance betweencomputational efficiency and reasoning depth. To save re-
sources, we compress the GENmodels tobfloat16.
4.1. Analysis of Design Choices
To verify the faithfulness of our approach and the validity
of our design choices, we carry out a number of small ex-
periments. These are performed on 200 randomly selected
samples from the MS-Marco v2.1 Q&A dataset (Nguyen
et al., 2016), considering 5 random context documents for
each query. The experiments are performed on 8 NVIDIA
RTX A5500 GPUs, each with 24GB of memory.
We measure faithfulness as the Area Inside the Perturbation
Curves (AIPC) via input perturbation. Since the query texts
are naturally short, and perturbing a single token can easily
distort the meaning of the whole text independent of the
token’s impact on the decision, we test faithfulness only on
the context documents. However, as the applied methods
are analogous for both contexts and queries, we argue that
the results are transferable. Appendix D details the process.
4.2. Experiments
We perform our main analysis on the following two datasets:
TREC CAsT 2019 (TC):a conversational IR benchmark
(38,636,520 texts) composed of MS-MARCO (Nguyen
et al., 2016), TREC CAR (Dietz et al., 2017), and WAPO,
with evaluation topics and human relevance judgments.
FoodSafeSum (FSS):a dataset in the food safety domain
with 124k documents and 133 evaluation topics with human-
annotated document relevancy; this dataset is not publicly
available due to copyright constraints and is therefore a good
effort to test the LLMs on previously unseen data. A subset
of the dataset was analysed by Bakagianni et al. (2025).
For each dataset, we construct flat FAISS (Douze et al.,
2024) indexes over document embeddings and retrieve the
top-10 documents per query. The documents are provided
to the GENunder two prompt configurations:(i)preserving
retrieval rank order, and(ii)randomly shuffling documents,
enabling analysis of sensitivity to document ordering. The
experiments conducted on these datasets are performed on
an Intel(R) Xeon(R) Platinum 8480CL with 8 NVIDIA
H100, each with 80GB of memory.
5. Results
We report the results of our experiments in the following
three sections:§5.1motivates the design choices of our
proposed RAG-E framework;§5.2presents an exploratory
analysis of RAG explanations using RAG-E (RQ1); and§5.3
empirically evaluates the utility of our WARG metric (RQ2).
5

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Table 1.Faithfulness in terms of AIPC (higher is better) for
different baselines Φ0of IG (with L=100 ). Bold scores mark the
most faithful baseline per RET. 95% confidence intervals (over
1000 bootstrap samples) are reported as [lower, upper].
RET Baselines (Φ0)
0[mask] [unk] [pad]
DRAGON 0.45[0.44, 0.46]0.46[0.45, 0.47]0.50[0.49, 0.51]0.41[0.39, 0.42]
Arctic Embed 2 0.68[0.66, 0.71]0.61[0.59, 0.64]0.73[0.70, 0.76]0.67[0.65, 0.69]
5.1. Empirical Analysis of Design Choices
Retriever Design Choices:To select a useful baseline
for the IG-based saliency values for the RET, we compare
replacing the embeddings of non-special tokens of the trans-
former input with the following values (see Fig. 3):(i)zeros
(discarding the positional embeddings),(ii)the [mask]
token,(iii)the [pad] token (suggested by the IG paper),
and(iv)the [unk] token, all embedded at the input posi-
tion. As shown in Tab. 1, a baseline replacing non-special
tokens with the model’s [unk] token clearly outperforms
the other choices. Further tests of IG on the RETcompo-
nent are reported in Appendix D. Specifically, we find that
using L= 100 integration steps sufficiently approximates
the integral and that IG is more faithful than other explain-
ability methods. Based on these findings, we apply IG with
a[unk]baseline andL= 100on RETin this paper.
5.86.06.26.4MSE
a)
567
b)
a)
50 100 150 200
Number of MC samples0.100.150.202
c)
10 15 20 25 30
N0.00.51.01.5
d)
c)
K
MC
K (compl.)
MC (compl.)
PMC (compl.)
Figure 4.MSE compared to precise SV [plots a)& b)] and vari-
anceσ2over 10 repetitions [plots c)& d)] ofKSHAP,MCSHAP,
andPMCSHAP for |D|= 5 (Right column:results for N= 20 ;
Left column:results for 200 MC samples).
Generator Design Choices:Fig. 4 presents the approx-
imation error (i.e., MSE), and reproducibility (i.e., vari-
ance σ2) of nativeKSHAP,MCSHAP, and our proposed
PMCSHAP. While the highest stability is achieved by
KSHAP with complementary sampling, we findPMCSHAP
the better choice, as it achieves the best approximation of
SV while displaying sufficiently low variance, especially
for high numbers of MC samples and a mediumN.
Further experiments reported in Appendix C show that(P)MCSHAP approximations are statistically significantly
closer to the true SV than nativeKSHAP under identical
sampling conditions. Nevertheless, none of the tested ap-
proaches outperforms the others in terms of faithfulness (see
Appendix D). In conclusion, we applyPMCSHAP with 200
MC samples to explain GEN.
5.2. Exploratory Analysis of RAG Attributions
To answerRQ1, we conduct an exploratory analysis of RET
and GENattributions. Overall, we find that even dense RET
rely on keyword-matching to retrieve documents while GEN
ones show an inherent primacy bias.
Retrieval Analysis:For the retrieval phase, we analyse
token-level attributions to understand which parts of the
query and documents drive the retrieval score. We aggre-
gated these IG scores across all queries to identify global
trends in feature importance. Given that this publication
is primarily concerned with explainability, we only report
retrieval performance in Appendix E for completeness.
We conduct a Part-of-Speech (POS)-based grammatical
analysis of queries and documents to examine how dif-
ferent parts of speech influence the retrieval phase. Our
analysis reveals that the retrieval mechanism relies heav-
ily on content-bearing Nouns (NOUN) and Proper Nouns
(PROPN), which together make up more than a third of the
top-30% attributed words. In comparison, other POS tags
make up around 10% or less each (see Appendix G). This
indicates that RETmatches entities and key concepts rather
than structural or function words. This is also supported by
qualitative inspection of the attributions (see Appendix I).
We also investigate the role of exact term matching. Our
analysis reveals a strong overlap of top-attributed tokens
in the retrieved documents and the query terms (see Ap-
pendix I). We observed that for the TC dataset, between
65.23% (Arctic Embed 2) and 78.23% (DRAGON) of the
query tokens appear among the top 50% attributed tokens
in the documents. For the FSS dataset, we observe simi-
larly high overlaps. When calculating the attributions using
DRAGON, we have a 78.18% overlap, while for Arctic
Embed v2, we observe 71.06% overlapping tokens. These
results reinforce our claim that despite the use of dense em-
beddings, the models still strongly rely on keyword overlap
between queries and documents.
Generation Analysis:We test generation with our frame-
work usingLlama 3.1 8BandGemma 3 12B. We focus on
stability of document attribution with respect to ordering.
We compare document attributions under ranked (according
to RET) and shuffled document orders in the prompt. Under
a generic prompting setup, a document’s importance should
depend on its content rather than its position. Note that in
6

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
our case, the prompt explicitly specifies the ranked nature
of the provided documents, which is expected to influence
the attributions assigned by the models (see Appendix B).
To measure stability, we compute the mean attribution score
for each document included in the prompt across all tested
configurations: ranked (according to RET), deduplicated
(for FSS), and shuffled document order.
Figure 5.Attribution Instability on FSS (left) and TC (right).
As shown in Fig. 5, both models tend to attribute more to
the first documents. While Gemma is more stable on the
FSS dataset, Llama exhibits stronger attributions for the
top-ranked documents even after shuffling them. Notably,
we still see this primacy bias in a small experiment using
a prompt not stating a specific ordering of documents (see
Appendix B). This suggests an inherent preference of infor-
mation based on its position in the prompt. When observing
the stability in the TC dataset, we have similar curves for
the original case and the case with shuffled documents, in-
dicating that preference is not tied to the actual content of
the documents. As before, both models tend to prioritize
the prompt information regarding the ranked nature of the
documents over the actual content, even though attribution
decreases on the first documents when shuffled.
Failure Mode Analysis (Wasted Retrieval and Noise Dis-
traction):To analyse consequences of the different inher-
ent behaviours of RETand GEN, indicated by the above
results, we quantify two specific failure modes:(i) Wasted
Retrievaloccurs when the top-ranked retrieved document
(RETRank 0) receives GENRank >2, indicating that the
RET’s most relevant document was largely ignored during
generation.(ii) Noise Distractionoccurs when GENassigns
its highest importance (GENRank 0) to a document with
RETRank >2, suggesting that the model was distracted
by content deemed less relevant by RET. The results are
shown in Table 2. Specifically, we seeWasted Retrievalin
47.4% (Llama on TC) to 66.7% (Gemma on FSS) of the
cases, andNoise Distractionin 48.1% (Llama on FSS) to
65.9% (Gemma on FSS) of the cases. While shuffling the
documents before generation increases these values, remov-
ing duplicates reduces them as it focuses attribution to fewerTable 2.Failure rates quantifying disagreement: “Wasted Retrieval”
(ignoring rank 0) and “Noise Distraction” (focusing on rank >2).
Model Dataset Condition Wasted Ret. (%) Noise Dist. (%)
Gemma FSS Orig 66.7 65.9
Gemma FSS ShuffOrig 63.975.2
Gemma FSS NoDup 64.7 71.4
Gemma FSS ShuffNoDup69.969.9
Gemma TC Orig 51.4 56.1
Gemma TC ShuffOrig 61.8 63.0
Llama FSS Orig 53.4 48.1
Llama FSS ShuffOrig 64.7 61.7
Llama FSS NoDup 48.9 49.6
Llama FSS ShuffNoDup 53.4 59.4
Llama TC Orig 47.4 49.1
Llama TC ShuffOrig 55.5 68.8
documents. These high values motivate further analysis into
quantification of the disagreement using WARG.
5.3. Verification of the WARG Metric
To assess the capability of our proposed WARG to quantify
the alignment between the RET’s and the GEN’s relevance
signals (RQ2), we compare it to standard Spearman corre-
lation. As can be seen in Table 3, the alignment between
the RETand the GENis low across all models and datasets.
In the standardOriginalcondition, Spearman correlation
peaks at modest values ( 0.255 for Gemma on TC) and drops
to zero for Gemma on the FSS dataset. This aligns with the
high WARG values, indicating high distance between the
order provided by RETand attributions of GEN.
We compareOriginalandNo Duplicatesfor the FSS dataset.
Since this data is composed of regulatory data that may be
identical or nearly identical, removing duplicate documents
should reduce noise, allowing the model to better focus
on important documents. This assumption is supported
by the WARG values, which show lower disagreement for
both models when compared with theOriginalretrieved
documents. Conversely, one would expect that shuffling the
order of the documents increases the gap between RETand
GEN. This is also visible in Table 3, taking into account
values ofpthat privilege the top documents.
Llama consistently exhibits higher agreement scores with
RETthan Gemma, with a Spearman correlation of 0.241
against 0.003 , obtained on the FSS dataset with no dupli-
cates. This suggests that Llama may be slightly more capa-
ble of identifying and attending to relevant content within
the context window, whereas Gemma’s attention is either
more diffuse or more strictly driven by positional heuristics.
This assumption aligns with the fact that while Llama keeps
higher WARG values for the shuffled version compared to
the correctly ordered documents when varying p, Gemma
shows closer values when focusing more on the deep list of
retrieved documents, i.e., higherpvalues.
7

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Table 3.RET-GENagreement metrics. We report the values with
the associated std for the Arctic Embed 2 RET.
Model Condition p=0.5 p=0.6 p=0.7 p=0.8 p=0.9 Spearman
Gemma (FSS) No Duplicates 0.774 0.734 0.687 0.655 0.702 0.003
Gemma (FSS) Original 0.813 0.767 0.711 0.665 0.699 -0.008
Gemma (FSS) Shuffled No Duplicates 0.784 0.739 0.689 0.655 0.702 0.010
Gemma (FSS) Shuffled Original 0.809 0.762 0.706 0.661 0.696 -0.010
Gemma (TC) Original 0.642 0.611 0.575 0.557 0.634 0.255
Gemma (TC) Shuffled Original 0.724 0.685 0.638 0.606 0.662 0.183
Llama (FSS) No Duplicates 0.623 0.595 0.565 0.558 0.644 0.241
Llama (FSS) Original 0.665 0.636 0.602 0.584 0.653 0.117
Llama (FSS) Shuffled No Duplicates 0.700 0.667 0.631 0.612 0.679 0.075
Llama (FSS) Shuffled Original 0.750 0.714 0.670 0.637 0.685 0.016
Llama (TC) Original 0.659 0.626 0.589 0.570 0.642 0.214
Llama (TC) Shuffled Original 0.764 0.719 0.667 0.628 0.676 0.114
In Appendix H, we report a comparison with the state-of-
the-art: ContextCite (Cohen-Wang et al., 2024).
6. Related Work & Discussion
Apart from a short paper by Sudhi et al. (2025), there is,
to the best of our knowledge, no peer-reviewed work on
end-to-end RAG explainability. The authors propose and
test an extrinsic method comparing RETand GENoutputs
after leave-one-out perturbations of the inputs and find their
explanations plausible in different user studies. However, it
remains unclear how exactly the comparison of outputs is
done, while evaluation of faithfulness is missing.
Explaining the RETcomponent in isolation has been studied
more extensively in literature. Zhuang et al. (2021) propose
aninterpretable-by-designranking model that could be ap-
plied in a RAG context. We argue, however, that limiting
oneself to a single RETarchitecture is not future-proof.
Fernando et al. (2019) evaluate intrinsic and extrinsic ex-
plainability methods on a single neural retriever, analyzing
the resulting top-attributed terms and observing substan-
tial variability across methods, but without assessing their
individual faithfulness. In contrast, our RAG-E adopts at-
tribution methods with explicit theoretical guarantees (IG
for RETand SV for GEN). Moreover, we improve the repro-
ducibility of SV approximations by proposingPMCSHAP.
Recent work on explaining RAG outputs has largely focused
on the GENcomponent, proposing algorithmic approaches
to source attribution rather than relying on model-generated
citations alone. Cohen-Wang et al. (2024) introduce Con-
textCite, which attributes generation to context using a linear
surrogate model and can be seen as a extension of LIME
(Ribeiro et al., 2016) to RAG. MIRAGE (Qi et al., 2024)
identifies context-sensitive tokens by measuring changes in
generation probabilities under document removal and ap-
plies contrastive attribution scores (Yin & Neubig, 2022).
We ground our GENattribution directly in SV and their
KSHAP approximations. For a small number of context
documents, this allows us to compute exact, theoreticallyfounded attributions with manageable computational cost
(e.g., one batch of size 64 for |D|= 6 ). Although attribu-
tion precision degrades with larger context sizes (as also
the case for ContextCite) existing methods often trade the-
oretical grounding, and therefore attribution quality, for
usability (Lundberg & Lee, 2017).
In summary, existing explainability approaches for RAG
systems typically focus on either retrieval or generation in
isolation. Our work differs in explicitly targeting the full
RAG pipeline and in providing tools to analyse/quantify
alignment and information flow between RETand GEN.
Specifically, our WARG is, to the best of our knowledge,
the first metric assessing RET-GENalignment enabling di-
agnosis of whether retrieved evidence is actually used down-
stream, rather than merely made available.
7. Conclusions
We presented RAG-E, a novel framework for end-to-end
explainability and diagnosis of RAG pipelines. We adapted
gradient-based IG to explain RETs, establishing suitable
baseline embeddings, and providedPMCSHAP, a stabilized
variant ofKSHAP that enables reliable attribution for au-
toregressive GENmodels. Using RAG-E, we performed
exploratory analyses on two RETmodels,DRAGONand
Arctic Embed 2, and two GENmodels,Llama 3 8Band
Gemma 3 12B, to understand which parts of the inputs and
outputs are most influential for RETand GEN(RQ1). We
observed that while both GENmodels exhibit inherent pri-
macy bias, Llama is more influenced by the user prompt
when attributing to the individual documents. Gemma tends
to value the actual content over the order of the documents
provided in the prompt. A grammatical analysis revealed
that dense RETmodels primarily rely on content-bearing
nouns, with limited sensitivity to adpositions and syntactic
connectors, resulting in behaviour that closely resembles
keyword matching between queries and documents. Be-
yond these component-level insights, our analyses exposed
a systematic misalignment between RETand GENmodels.
This suggests that RAG performance should be understood
less as a property of individual components and more as an
alignment problem between RETand GEN. To quantify this
disagreement, we proposed and evaluated WARG, a metric
grounded in the assumption that top-ranked documents best
reflect user-relevant information (RQ2).
Overall, RAG-E provides a principled framework for ex-
plaining and auditing RAG pipelines by making information
flow across components explicit and measurable. Future
directions include scalability and extending RAG-E and
WARG to include rerankers. Finally, we plan to investigate
the use of WARG as an optimization signal, for example, in
query rewriting or reinforcement learning-based extensions
of RAG systems.
8

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Acknowledgements
This work has been partially supported by project MIS
5154714 of the National Recovery and Resilience Plan
Greece 2.0 funded by the European Union under the
NextGenerationEU Program. Funding for this research
has also been provided by the European Union’s Horizon
Europe research and innovation programme EFRA (Grant
Agreement Number 101093026). Views and opinions ex-
pressed are, however, those of the authors only and do not
necessarily reflect those of the European Union or Euro-
pean Commission-EU. Neither the European Union nor the
granting authority can be held responsible for them.⋆⋆⋆⋆⋆
⋆
⋆
⋆
⋆⋆⋆⋆
Impact Statement
RAG systems are increasingly deployed in applications that
support decision-making in high-stakes domains such as
healthcare and law (Amugongo et al., 2025; Pipitone &
Alami, 2024). This work contributes tools for auditing
and explaining RAG pipelines by making the interaction
between retrieval and generation explicit and measurable.
By quantifying retriever–generator misalignment and iden-
tifying systematic failure modes, our framework can sup-
port more reliable system design, post-hoc analysis, and
informed human oversight.
At the same time, we are aware that the methods proposed
in this paper rely on a certain level of technical understand-
ing to interpret. Untrained users may misinterpret results,
and, for example, confuse low WARG with a direct indi-
cation of factual correctness or truthfulness. Misuse of
such metrics could lead to overconfidence in aligned but
incorrect systems. Moreover, attribution-based explanations
may be misunderstood as fully causal by non-expert users,
underscoring the importance of careful interpretation and
complementary evaluation.
Overall, we view this work as enabling more transparent
and accountable use of RAG systems, while recognizing
that explainability tools must be applied judiciously and in
conjunction with domain expertise and external validation.
References
Abnar, S. and Zuidema, W. Quantifying attention flow in
transformers. InProceedings of ACL, pp. 4190–4197.
ACL, 2020.
Amugongo, L. M., Mascheroni, P., Brooks, S., Doering,
S., and Seidel, J. Retrieval augmented generation for
large language models in healthcare: A systematic review.
PLOS Digital Health, 4(6):1–33, 06 2025.
Angiulli, F., De Luca, F., Fassetti, F., and Nistic `o, S. LLiMe:enhancing text classifier explanations with large language
models.Machine Learning, 114(12):271, 2025.
Asai, A., Wu, Z., Wang, Y ., et al. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection. In
Proceedings of ICLR, 2024.
Bakagianni, J., Randl, K., Rocchietti, G., et al. FoodSafe-
Sum: Enabling natural language processing applications
for food safety document summarization and analysis. In
Findings of EMNLP, pp. 16786–16804. ACL, 2025.
Brown, A., Roman, M., and Devereux, B. A systematic
literature review of retrieval-augmented generation: Tech-
niques, metrics, and challenges.Big Data and Cognitive
Computing, 9(12), 2025. ISSN 2504-2289.
Cohen-Wang, B., Shah, H., Georgiev, K., and Madry, A.
ContextCite: attributing model generation to context. In
Advances in NeurIPS. Curran Associates Inc., 2024.
Covert, I. and Lee, S.-I. Improving kernelshap: Practical
shapley value estimation using linear regression. InPro-
ceedings of Machine Learning Research, volume 130, pp.
3457–3465. PMLR, 2021.
Dietz, L., Verma, M., Radlinski, F., and Craswell, N. TREC
complex answer retrieval overview. InProceedings of
TREC, pp. 13, 2017.
Douze, M., Guzhva, A., Deng, C., et al. The faiss library,
2024. Preprint at https://arxiv.org/abs/2401.
08281.
Edin, J., Motzfeldt, A. G., Christensen, C. L., et al. Nor-
malized AOPC: Fixing misleading faithfulness metrics
for feature attributions explainability. InProceedings of
ACL, pp. 1715–1730. ACL, 2025.
Fernando, Z. T., Singh, J., and Anand, A. A study on the
interpretability of neural retrieval models using deepshap.
InProceedings of SIGIR, pp. 1005–1008. ACM, 2019.
Grattafiori, A., Dubey, A., Jauhri, A., et al. The
llama 3 herd of models, 2024. Preprint at
https://arxiv.org/abs/2407.21783 . Hug-
gingface model:meta-llama/Llama-3.1-8B.
Izacard, G., Lewis, P. S. H., Lomeli, M., Hosseini, L.,
Petroni, F., Schick, T., Dwivedi-Yu, J., Joulin, A., Riedel,
S., and Grave, E. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine Learn-
ing Research, 24:251:1–251:43, 2023.
Jain, S. and Wallace, B. C. Attention is not Explanation. In
Proceedings of NACL, pp. 3543–3556. ACL, 2019.
9

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Kamath, A., Ferret, J., Pathak, S., et al. Gemma
3 technical report, 2025. Preprint at https:
//arxiv.org/abs/2503.19786 . Huggingface
model:google/gemma-3-12b-it.
Kuratomi, A., Lee, Z., Miliou, I., Lindgren, T., and Papa-
petrou, P. ORANGE: Opposite-label soRting for tAN-
Gent Explanations in heterogeneous spaces. InProceeed-
ings of DSAA, pp. 1–10. IEEE, 2023.
Lewis, P., Perez, E., Piktus, A., et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. InAd-
vances in NeurIPS, volume 33, pp. 9459–9474. Curran
Associates Inc., 2020.
Lin, S.-C., Asai, A., Li, M., et al. How to
train your DRAGON: Diverse augmentation to-
wards generalizable dense retrieval, 2023. Preprint
athttps://arxiv.org/abs/2302.07452 . Hug-
gingface model:facebook/
dragon-plus-[context|query]-encoder.
Liu, N. F., Lin, K., Hewitt, J., et al. Lost in the middle: How
language models use long contexts.TACL, 12:157–173,
2024.
Liu, S., Le, F., Chakraborty, S., and Abdelzaher, T. On ex-
ploring attention-based explanation for transformer mod-
els in text classification. InProceedings of Big Data, pp.
1193–1203, 2021.
Lundberg, S. M. and Lee, S.-I. A unified approach to in-
terpreting model predictions. InAdvances in NeurIPS,
volume 30, pp. 4765–4774. Curran Associates, Inc., 2017.
Mitchell, R., Frank, E., and Holmes, G. GPUTreeShap:
massively parallel exact calculation of SHAP scores for
tree ensembles.PeerJ Computer Science, 8:e880, 2022.
doi: 10.7717/peerj-cs.880.
Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S.,
Majumder, R., and Deng, L. MS MARCO: A human gen-
erated machine reading comprehension dataset. InCEUR
Workshop Proceedings, volume 1773. CEUR-WS.org,
2016.
Pipitone, N. and Alami, G. H. Legalbench-rag: A bench-
mark for retrieval-augmented generation in the legal do-
main, 2024. Preprint at https://arxiv.org/abs/
2408.10343.
Qi, J., Sarti, G., Fern ´andez, R., and Bisazza, A.
Model Internals-based answer attribution for trustwor-
thy Retrieval-Augmented GEneration. InProceedings of
EMNLP, pp. 6037–6053. ACL, 2024.
Randl, K., Pavlopoulos, J., Henriksson, A., and Lindgren, T.
Mind the gap: from plausible to valid self-explanationsin large language models.Machine Learning, 114(10):
220, 2025.
Ribeiro, M. T., Singh, S., and Guestrin, C. ”why should i
trust you?”: Explaining the predictions of any classifier.
InProceedings of KDD, pp. 1135–1144, 2016.
Shapley, L. S. 17. a value for n-person games. In Kuhn,
H. W. and Tucker, A. W. (eds.),Contributions to the
Theory of Games, Volume II, pp. 307–318. Princeton
University Press, 1953.
Shrikumar, A., Greenside, P., and Kundaje, A. Learn-
ing important features through propagating activation
differences. InProceedings of ICML, volume 70, pp.
3145–3153, 2017.
Sudhi, V ., Bhat, S. R., Rudat, M., et al. Towards end-
to-end model-agnostic explanations for rag systems,
2025. Preprint at https://arxiv.org/abs/2509.
07620.
Sundararajan, M., Taly, A., and Yan, Q. Axiomatic at-
tribution for deep networks. InProceedings of ICML,
volume 70, pp. 3319–3328, 2017.
Tan, Z., Tian, Y ., and Li, J. Glime: General, stable and local
lime explanation. InAdvances in NeurIPS, volume 36,
pp. 36250–36277. Curran Associates, Inc., 2023.
Webber, W., Moffat, A., and Zobel, J. A similarity measure
for indefinite rankings.ACM Trans. Inf. Syst., 28(4),
2010.
Yin, K. and Neubig, G. Interpreting language models with
contrastive explanations. InProceedings of EMNLP, pp.
184–198. ACL, 2022.
Yu, P., Merrick, L., Nuti, G., and Campos, D. F. Arctic-
embed 2.0: Multilingual retrieval without compromise.
InProceedings of COLM, October 2025. Huggingface
model:Snowflake/
snowflake-arctic-embed-l-v2.0.
Zhuang, H., Wang, X., Bendersky, M., et al. Interpretable
ranking with generalized additive models. InProceedings
of WSDM, 2021.
10

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
A. Extended Background
Generally, methods explaining ML models are commonly separated intointrinsicmethods, deriving explanations based
on the internal state of the model, andextrinsicormodel agnosticmethods, which statistically infer explanations from
input-to-output-relationships without considering model internals.
Earlyintrinsic methodstailored to transformer models were eitherattention-basedorgradient-basedmethods.Attention-
based methodsuse the fact that the self-attention weights of transformers can be interpreted as weights of how a specific
input token impacts the output. They range from simply using the raw attention weights of the last layer as an explanation,
to more wholistic approaches tracking attention through the whole transformer (Abnar & Zuidema, 2020). Finally, there
are also hybrid approaches that consider the gradient of the output with regard to the attention weight (Liu et al., 2021).
Nevertheless, Jain & Wallace (2019) criticise at least the use of raw attention weights as explanations to overestimate their
correlation to the output.Gradient-based methodsevaluate the gradient∂yj
∂xi
xi=¯xito get a linear approximation of the
impact of the ithelement of the input on the jthelement of the output. Simple methods include directly using the gradient
as an explanation or multiplying it with the input (Shrikumar et al., 2017). Nevertheless, while optimal in the immediate
neighbourhood of ˆxi, the raw gradient is not necessarily a good approximation of the global function learned by the ML
model (Shrikumar et al., 2017; Sundararajan et al., 2017, see Figure 6). Sundararajan et al. (2017) solve this problem by
integrating gradients over the input dimension by proposing IG.
xiyj
f(xi) f(xi)
xiPOS
NEG∂yj
∂xi· xi
xi=xi
(a)Gradient as a linear approximation.
xiyj
f(xi) f(xi)
xiPOS
NEG (b)Non-gradient linear approximation.
Model 
FunctionDecision 
BoundaryLinear 
Approx.
Figure 6.Gradient-based saliency and its caveats. While the gradient is an optimal local linear approximation in the point of the input ¯xi,
this is not necessarily the case globally or with regard to the intersect of model function and decision boundary.
Among the most widely usedextrinsic methodsfor LMs are Local Interpretable Model-agnostic Explanations (LIME)
by Ribeiro et al. (2016) and SHapley Additive exPlanations (SHAP) by Lundberg & Lee (2017). LIME trains a surrogate
Logistic Regression classifier on tuples (¯x′,¯y′), with ¯x′randomly sampled in the close vicinity of ¯x, and ¯y′=f(¯x′)by
perturbing elements in ¯x. The coefficients of this classifier are used as linear attribution scores. Recent extensions of
this method focus mostly on improving the generation ¯x′for different types of data (Kuratomi et al., 2023; Tan et al.,
2023; Angiulli et al., 2025).KSHAP (Lundberg & Lee, 2017) adapts the LIME approach to approximate SV from game
theory (Shapley, 1953) by deriving a loss function for training the surrogate model which satisfies theadditivity,missingness,
andconsistencyconstraints of SVs. Extensions of the SHAP approach usually focus on creating faster, intrinsic methods
that also satisfy these constraints for more specific classifiers (Mitchell et al., 2022). Nevertheless, (Covert & Lee, 2021)
test different methods for sampling the input-output pairs necessary to train the surrogate model. They find that sampling
complementary pairs of feature sets at the input and their respective outputs improves the stability ofKSHAP.
11

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
B. Generation Prompts
Fig. 7 shows the formats for the prompts used to generate answers in our experiments and ablations. These correspond to the
“create prompt(q,D) ” functions referenced in the main text. In Llama we include the instructions in the system prompt
and provide query and context documents tin the user prompt. Since Gemma does not support system prompts, we include
everything in the user prompt.
SYSTEM:Use the following retrieved documents, ranked from highest
to lowest relevance, to answer the user’s query.
Be thorough and accurate, and cite documents when useful.
Keep the answer under 200 words.
USER:Document 1: [...]
Document 2: [...]
Document 3: [...]
Document 4: [...]
Document 5: [...]
Query: [...]
MODEL:...
(a)LlamaUSER:Use the following retrieved documents, ranked from highest
to lowest relevance, to answer the user’s query.
Be thorough and accurate, and cite documents when useful.
Keep the answer under 200 words.
Document 1: [...]
Document 2: [...]
Document 3: [...]
Document 4: [...]
Document 5: [...]
Query: [...]
MODEL:...
(b)Gemma
Figure 7.Prompt formats for the GENmodels. Grayed sections “[...]” are replaced by the respective content.
For the sake of completeness, we selected a subset of the two datasets of around 40% of the queries and tested a generic
prompt:Use the following documents to answer the user’s query. Be thorough and accurate, and cite documents when
useful. Keep the answer under 200 words.In this case, we completely removed the information regarding the ordered nature
of the documents to observe whether the models would distribute their attribution the same way or not. In Figure 8, we
report the attribution instability across the documents.
Figure 8.Attribution Instability on the FSS (left) and TC (right) Dataset with thegenericprompt.
In contrast to the Attribution Instability shown in Figure 5, we can observe here that the attributions given by the GENare
less stable than they were before. Regarding the FSS dataset, we can see that the curve for Gemma with theoriginalandno
duplicatesagainst their shuffled version is very similar, suggesting some kind of primacy bias for theoriginalsetup, while
in the case ofno duplicates, it seems that the GENignores the suggestion given by the RET. When observing Llama, the
curves are more unstable. When considering TC, we notice that Gemma has a flatter curve when shuffling the documents,
compared to a decreasing curve for theoriginalsetup. The results obtained by Llama suggest a higher primacy bias since
the differences between the curves for the original and shuffled setups are minimal, with the first being slightly more stable
than the second one.
12

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
C. Monte-Carlo stabilization for Kernel SHAP
p=0.9535
p=0.5090
p=0.2982
p=0.1778
p=0.0155
p=0.0089
p=0.0094
p=0.0044
p=0.0084
p=0.000350 100 150 200
0.4660.4680.470Pearson r
a)
p=0.1933
p=0.0042
p=0.0003
p=0.9685
p=1.000010 15 20 25 30
0.4000.4250.4500.475
b)a)
Kernel-SHAP
Monte Carlop=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
1.351.361.371.38MAE
c)
p=0.0151
p=0.0000
p=0.0000
p=0.0000
p=0.0000
1.41.51.6
d)
c)
p=0.0044
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
p=0.0000
50 100 150 200
Number of MC samples6.256.306.356.40MSE
e)
p=0.0429
p=0.0000
p=0.0000
p=0.0000
p=0.0000
10 15 20 25 30
N6.57.07.5
f)
e)
Figure 9.Mean Pearson r, MAE, and MSE comparingKSHAP andMCSHAP to precise SV over 200 random samples of MS-Marco.
Computed for |D|= 5 and nativeuniform sampling. The blue vertical numbers are the p-values for a paired Wilcoxon Signed Rank
test with alternate hypothesis that theMCSHAP metric is greater than (for Pearson- r) or less than (MAE & MSE) theKSHAP metric
(Right column:results forN= 20;Left column:results for 200 MC samples).
p=0.9465 | 0.9634
p=0.8829 | 0.7127
p=0.6419 | 0.6328
p=0.7438 | 0.6790
p=0.6546 | 0.5659
p=0.5403 | 0.5572
p=0.5384 | 0.6768
p=0.6180 | 0.4781
p=0.5287 | 0.5616
p=0.4708 | 0.514150 100 150 200
0.4650.4700.475Pearson r
a)
p=0.5282 | 0.4718
p=0.0984 | 0.1293
p=0.4708 | 0.5141
p=0.7570 | 0.2445
p=0.8291 | 0.085510 15 20 25 30
0.4000.4250.4500.475
b)a)
Kernel-SHAP
Monte Carlo
Paired Monte Carlop=0.1431 | 0.0054
p=0.0788 | 0.0006
p=0.0180 | 0.0005
p=0.0272 | 0.0003
p=0.0141 | 0.0002
p=0.0059 | 0.0003
p=0.0094 | 0.0003
p=0.0114 | 0.0001
p=0.0057 | 0.0001
p=0.0051 | 0.0001
1.301.321.34MAE
c)
p=0.4245 | 0.5755
p=0.0000 | 0.0000
p=0.0051 | 0.0001
p=0.0219 | 0.0000
p=0.0396 | 0.0000
1.251.301.351.40
d)
c)
p=0.1515 | 0.0113
p=0.1046 | 0.0031
p=0.0552 | 0.0024
p=0.0576 | 0.0020
p=0.0377 | 0.0012
p=0.0431 | 0.0025
p=0.0423 | 0.0021
p=0.0465 | 0.0013
p=0.0382 | 0.0013
p=0.0316 | 0.0012
50 100 150 200
Number of MC samples5.86.06.2MSE
e)
p=0.4202 | 0.5798
p=0.0001 | 0.0001
p=0.0316 | 0.0012
p=0.1031 | 0.0008
p=0.1744 | 0.0000
10 15 20 25 30
N5.56.06.5
f) e)
Figure 10.Pearson r, MAE, and MSE comparingKSHAP,MCSHAP, andPMCSHAP over 200 random samples of MS-Marco. Computed
for precise SV for |D|= 5 andcomplementary sampling. The blue/orange vertical numbers are the p-values for a paired Wilcoxon
Signed Rank test with alternate hypothesis that the (P)MCSHAP metric is greater than (for Pearson- r) or less than (MAE & MSE) the
KSHAP metric (Right column:results forN= 20;Left column:results for 200 MC samples).
13

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Figures 9 and 10 show comparisons ofKSHAP and (P)MCSHAP methods for the GEN. We present comparisons for
both native uniform sampling (Fig. 9) and complementary sampling (Fig. 10). A non-parametric Wilcoxon Signed Rank
test shows a significant improvement of uniformMCSHAP and complementaryPMCSHAP overKSHAP under identical
sampling for ≥40 MC samples at N= 20 or15≤N≤25 at 200 MC samples. For uniform sampling, we also see
significantly better correlation of SV andMCSHAP for these values.
D. Faithfulness Metric
We measure faithfulness as the Area Inside the Perturbation Curves (AIPC) via input perturbation. Concretely, we repeatedly
call the model while gradually masking input tokens according to their attributed importance until the input is completely
obscured. We perturb the input in both directions, removing tokens Most Relevant First (MoRF) and Least Relevant First
(LeRF). Afaithfulexplanation should trigger an early change in the MoRF setting, since highly influential tokens are masked
first, and only a late change in the LeRF setting, because initially unimportant tokens are removed and should not affect the
prediction much.
This metric is widely applied in literature but not standardized (Liu et al., 2021; Edin et al., 2025; Randl et al., 2025). We
compute AIPC per input sequence x, masking it from 0to all ntokens, one token at a time. Let mask dir(x, i) denote the
inputxwithitokens masked according to direction “dir”. We measure the area in between the perturbation curves
AIPC qry=Rn
i=0g(mask MoRF(q, i),d)−g(q,d)di−
Rn
i=0g(mask LeRF(q, i),d)−g(q,d)di(13)
for the query and
AIPC ctx=Rn
i=0g(q,d)−g(d,mask MoRF(d, i))di−
Rn
i=0g(q,d)−g(d,mask LeRF(d, i))di(14)
for context documents. For the RETwe choose g(q,d) =s ret(q,d) and for the GEN g(q,d) is the model output. We then
min-max-normalize per sample and report the mean AIPC over all inputs. Consecutivelly, AIPC values theoretically range
between 0(not faithful) and 1(maximally faithful). Note that the realistic maximum is model dependent and always less
than1.
RETExplanations:We compare the faithfulness of our IG RETexplanations to the raw gradient
Grad:β i=∂f(x)
∂xi
,
gradient ×input
GradIn:β i=∂f(x)
∂xi·xi
(Shrikumar et al., 2017), and Attention Gradient (AGrad)(Liu et al., 2021) as
baselines. As both eqry(·)andectx(·)are encoder-only transformers y=f(Φ(x)) atop an embedding Φ(x) =Wx+p
(where pis a positional embedding), this means we have to also calculate the gradient over Φ(x) which is not supported by
PyTorch’sautograd. We therefore calculate the gradient to the input manually:
∂e(x)
∂x=∂f(Φ(x))
∂Φ(x)
x=¯x|{z }
computed byautograd·W.(15)
Since the query texts are naturally short in nature, and perturbing a single token can easily distort the meaning of the whole
text, independent of the tokens’ impact on the decision, we test faithfulness only on the context documents. However, as the
applied IG method is identical for contexts and query, we argue that the results are transferable. The results of this test are
shown in Tab. 4. For both tested RETs, IG performs best. Although, surprisingly, AGrad is equally faithful as IG for the
DRAGON RET, IG remains the better choice overall.
To test the design choices we use for the IG explanations, we ablated the number of Riemann integration steps Lin terms of
faithfulness and additivity ratio. Our results, presented in Tab. 5, suggest that using L= 100 steps sufficiently approximates
the integral, as the attributions sum up to sufficiently more than 90% of the output, and faithfulness on the context attributions
plateaus.
14

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Table 4.RETfaithfulness in terms of AIPC (higher is better) for different explanation methods. Bold scores mark the most faithful
explainability method per RETmodel. 95% confidence intervals (computed over 1000 bootstrap samples) are reported as [lower, upper].
RET Methods
RND Grad GradIn AGrad IG
DRAGON0.00 0.21 0.100.50 0.50
[-0.01, 0.01] [0.19, 0.22] [0.09, 0.12][0.49, 0.51] [0.49, 0.51]
Arctic Embed 20.00 -0.18 0.23 0.640.73
[-0.01, 0.02] [-0.21, -0.15] [0.21, 0.25] [0.62, 0.66][0.70, 0.76]
Table 5.Faithfulness in terms of AIPC (higher is better) for different integration steps Lof IG ( Φ0fixed to [pad] ), with additivity ratios
reported for query and context. Bold scores mark the most faithful IG setting per RET. 95% confidence intervals (computed over 1000
bootstrap samples) are reported as [lower, upper].
RET/ Metric Integration Steps (L)
L=10L=50L=100
DRAGONAIPC 0.40[0.39, 0.41]0.41[0.39, 0.42]0.41[0.39, 0.42]
Pn
i=1βi,j
¯yj−¯y0
jquery 0.84[0.02, 1.99]1.00[0.93, 1.08]0.94[0.88, 0.99]
context 0.89[0.88, 0.91]0.98[0.97, 0.98]0.99[0.99, 0.99]
ArcticAIPC 0.65[0.62, 0.67]0.67[0.65, 0.69]0.67[0.65, 0.69]
Embed 2Pn
i=1βi,j
¯yj−¯y0
jquery 0.91[0.87, 0.96]0.98[0.98, 0.99]0.99[0.99, 0.99]
context 0.77[0.65, 0.86]0.99[0.90, 1.12]0.98[0.95, 1.01]
GENExplanations:In order to verify the faithfulness of our GENexplanations, we compute AIPC for nativeKSHAP,
MCSHAP, and our proposedPMCSHAP for native uniform sampling as well as complementary sampling (Covert & Lee,
2021) over differentsample sizesNand report the results in Tab. 6. The results show comparable faithfulness overall.
Table 6.Faithfulness in terms of AIPC (higher is better) for differentKSHAP extensions and |D|= 5 . 95% confidence intervals
(computed over 1000 bootstrap samples) are reported as [lower, upper]. The columns “Random” and “Precise” show the faithfulness of
randomized and precise SV attributions, and signify theoretical bounds for faithfulness.
Method Sampling AIPC
Random N=10N=15N=20N=25N=30 Precise
KSHAP unif.
0.00 [−0.01,0.01]0.51[0.49, 0.53]0.55[0.53, 0.57]0.56[0.54, 0.59]0.57[0.55, 0.60]0.58[0.56, 0.60]
0.58 [0.56,0.60]MCSHAP unif. 0.51[0.49, 0.53]0.55[0.53, 0.57]0.56[0.54, 0.59]0.57[0.55, 0.60]0.58[0.56, 0.60]
KSHAP compl. 0.49[0.47, 0.51]0.55[0.52, 0.57]0.57[0.55, 0.60]0.58[0.55, 0.60]0.58[0.56, 0.61]
MCSHAP compl. 0.50[0.48, 0.52]0.54[0.52, 0.57]0.57[0.55, 0.59]0.57[0.55, 0.60]0.58[0.56, 0.60]
PMCSHAP compl. 0.49[0.47, 0.51]0.55[0.53, 0.57]0.57[0.54, 0.59]0.57[0.55, 0.60]0.58[0.56, 0.60]
15

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
E. Retrieval Performance
To validate our retrieval component, we evaluated the performance of theDRAGONandArctic Embed 2embedders on the
FSS and TC datasets. In Tab. 7 we report the NDCG@3, and Precision@1.
Table 7.Retrieval Performances. We report here Precision@1 (P@1) and Normalize Discounted Cumulative Gain for the first three
documents (NDCG@3)
Configuration P@1 NDCG@3
FSS (Arctic Embed 2) 0.221 0.155
FSS (DRAGON) 0.183 0.116
TC (Arctic Embed 2) 0.723 0.475
TC (DRAGON) 0.699 0.441
In this phase, we report only fine-grained metrics given the fact that our goal is not to evaluate the retrieval phase but to
assess the alignment between the RETand the GEN.
F. Results with DRAGON RET
Table 8.WARG-Rank Sensitivity to p and Spearman Correlation (Dragon Retriever)
Model Condition p=0.5 p=0.6 p=0.7 p=0.8 p=0.9 Spearman
Gemma (FSS) No Duplicates 0.778 0.733 0.681 0.641 0.685 0.055
Gemma (FSS) Original 0.804 0.761 0.708 0.664 0.700 -0.024
Gemma (TREC) Original 0.655 0.617 0.576 0.554 0.630 0.292
Gemma (TREC) Shuffled Original 0.728 0.683 0.633 0.599 0.656 0.215
Llama (FSS) No Duplicates 0.638 0.613 0.583 0.569 0.645 0.170
Llama (FSS) Original 0.647 0.616 0.581 0.565 0.640 0.190
Llama (FSS) Shuffled No Duplicates 0.701 0.667 0.627 0.603 0.665 0.082
Llama (FSS) Shuffled Original 0.740 0.701 0.654 0.620 0.672 0.106
Llama (TREC) Original 0.633 0.601 0.567 0.552 0.631 0.234
Llama (TREC) Shuffled Original 0.697 0.664 0.625 0.600 0.661 0.126
16

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
G. POS Analysis
In this section we report some analysis regarding the attribution given by the models to the different part of speech. We
analysed the distribution of attribution weights across Part-of-Speech (POS) tags and POS bigrams (word pairs) for both
GENs and RETs, focusing on the top 30% of attributed tokens (Figures 11 and 12). Our analysis reveals several key
similarities and distinctions in how these models attend to linguistic features. To be able to compute so, we first reconstructed
the words from the tokens and successively summed the attributions outputted by the models.
Across all four models, NOUNs consistently receive the highest attribution mass. This indicates a shared reliance on
substantive content words as the primary elements of meaning and relevance. Furthermore, NOUN + NOUN is overall the
most significant bigram, which highlights the importance of nouns and nominal phrases in both generation and retrieval
contexts. While both GENs prioritize nouns, their secondary focuses diverge significantly. Gemma exhibits a high sensitivity
to punctuation and structural markers. PUNCT is the second most attributed tag, and NOUN + PUNCT values are prominent.
This might suggest that Gemma’s attention mechanism relies heavily on sentence boundaries and delimiters to organize
context. On the other hand, Llama displays a more syntactic focus, with high attribution to VERBs and function words
like ADP (adpositions). Its bigrams, such as NOUN + ADP and DET + NOUN, reflect a deeper engagement with the
grammatical structure and prepositional relationships within the text, rather than just isolated keywords or delimiters.
(a)Unigram POS Attribution
 (b)Bigram POS Attribution
Figure 11.Unigram and Bigram Attribution Mass for LLama and Gemma
RETComparison: Snowflake vs. DragonThe RETs show a stronger alignment with entity-centric processing compared
to the GENs, but with nuanced differences. Arctic-Embed (Snowflake) shows a distinct preference for PROPN (Proper
Nouns), which ranks second only to common nouns. Its top bigrams, like PROPN + NOUN , suggest it functions closer to
a traditional keyword-based RET, heavily prioritizing named entities and specific terminology to match queries. Dragon
(RoBERTa-based) shares the entity focus (high PROPN) but also places significant weight on ADJ (Adjectives) and PRON
(Pronouns). This suggests a more semantic understanding of the query, attending to descriptive qualifiers and anaphoric
references (e.g., PRON + AUX ) which are typical of dense retrieval models that capture ”meaning” beyond exact keyword
matches.
17

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
(a)Unigram POS Attribution
 (b)Bigram POS Attribution
Figure 12.POS attribution forArctic Embed 2andDragon.
GENvs. RETA broad distinction emerges between the two classes of models. On the one hand, GENs (especially Llama)
distribute attention more broadly across syntactic elements ( VERB ,ADP,DET) required to construct coherent sentences and
follow narrative flow. On the other hand, RETs concentrate their mass more narrowly on content-bearing classes ( NOUN ,
PROPN,ADJ), filtering out much of the connectors to maximize relevance attribution.
H. Comparison with ContextCite
We compared our perturbation-based attribution method with ContextCite, an attention-based attribution baseline. Table 9
summarizes the agreement (WARG-0.9) and correlation (Spearman) metrics for both methods.
Our Method (Perturb) ContextCite (Attn)
Model (GEN) Dataset WARG-0.9↓Spearman↑WARG-0.9↓Spearman↑
Gemma FSS 0.75 0.010.12-0.04
Gemma TC 0.720.200.67 -0.10
Llama FSS0.72 0.191.65 -0.06
Llama TC 0.710.27 0.620.09
Table 9.Comparison of RET-GENAgreement between our Perturbation-based method and ContextCite. WARG-0.9 (Weighted Agreement
Rank Gap) favors lower values; Spearman correlation favors higher values.
The results reveal distinct behaviors. On the FSS dataset with Gemma, ContextCite yields an exceptionally low WARG
(0.12), implying the attention distribution closely mirrors the retrieval ranking mass. However, this mechanical alignment
does not translate to rank correlation (Spearman ≈0), suggesting the weights match but the specific ordering might not.
Conversely, on Llama with FSS, ContextCite shows very high disagreement (WARG 1.65).
Our perturbation-based method produces more consistent agreement scores (WARG ≈0.7 ) across models and datasets, and
generally higher Spearman correlations, particularly on TC (0.20–0.27). This suggests that while attention mechanisms
(ContextCite) may sometimes align linearly with input order (due to positional bias in attention), perturbation-based
causal analysis better captures the relationship between retrieval relevance and generation utility. The high rates of “noise
distraction” detected by our method reflect this causal divergence: the GENoften functionally relies on documents that the
RET(and potentially simple attention maps) did not prioritize.
18

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
I. Comparative Analysis of Attribution Patterns
In this section, we showcase two examples for each dataset, highlighting the extremes of our WARG attribution for both
datasets. We report, for every query, the attribution for the query and the retrieved documents, as well as the impact of each
document on the generated text (Assistant). In Table 10 we also show the percentage of query tokens that appear among the
top 50% most attributed tokens as discussed in §5.
Table 10.Percentage of query tokens appearing among the top 50% attributed tokens in retrieved documents.
Dataset Model (RET) Query Overlap (%)
TCArctic Embed 2 65.23%
DRAGON 78.23%
FSSArctic Embed 2 71.06%
DRAGON 78.18%
I.1. Case Study 1: Low Agreement
In Figures 13 and 14, we show the attribution patterns for two queries that exhibited low agreement between the retriever
and the generator.
As we can observe, in both cases, the most meaningful words in the query are also among the most attributed parts of the
documents (food,flavor, andagentsfor the FSS example; andpop,music, andeducationfor the TC example). Interestingly
enough, the RETalso highly attributes semantically related words (e.g.,lesson based,popular,pedagogy).
When observing the GENattributions, we can see that all of the documents contributed to creating the answer when
considering Figure 13 with Documents 9 and 10 contributing more than some would expect from the retriever rank. When
focusing on the TC example, we clearly see that documents 5,6,7, and 9 contribute the most to the generated text in
comparison to the first ranked documents. This could be due to the similarity (complete overlapping in some case) among
the retrieved documents
I.2. Case Study 2: High Agreement
Similarly to Case Study 1, in Figures 15 and 16, we show the attributions for both the RETand the GEN. Similarly to what
we observed before for the RETside, we observe strong attributions for words that appear in the query. This is particularly
evident in figure 16 where the query is short and contains less meaningful words.
When observing the attributions for the generated texts, we can observe again a similar behaviour. When looking at the
results computed on the FSS dataset, we show a higher variance in attribution, but we can see that many of the most
attributed words (intense color) belong to the first documents. On the other hand, when looking at Figure 16 regarding the
TC dataset, we can clearly see that Documents 1 and 3 are among the most attributed.
19

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Query:What arethedifferent types offlavor enhancers andtheir functions infood products, andhow dothey compare toother food additives likeemulsifiers andbulking agents?
Document 1:”...However, substances should notbeconsidered asfood additives when they areused forthepurpose ofimparting flavour and/or taste orfornutritional purposes,
such assaltreplacers, vitamins andminerals. ...”Finally, Article
Document 2:”...However, substances should notbeconsidered asfood additives when they areused forthepurpose ofimparting flavour and/or taste orfornutritional purposes,
such assaltreplacers, vitamins andminerals. ...”Finally, Article
Document 3:(5)Curing andpickling agents :Substances imparting aunique flavor and/or color toafood, usually producing anincrease inshelf lifestability.
(6)Dough strengtheners :Substances used tomodify starch andgluten,
Document 4:(5)Curing andpickling agents :Substances imparting aunique flavor and/or color toafood, usually producing anincrease inshelf lifestability.
(6)Dough strengtheners :Substances used tomodify starch andgluten,
Document 5:(10) Firming agents: Substances added toprecipitate residual pectin, thus strengthening thesupporting tissue andpreventing itscollapse during processing.
(11) Flavor enhancers: Substances added tosupplement, enhance, ormodify theoriginal taste and/or aroma
Document 6:—Strength boosters —Firming agents —Substances thatcreate ormaintain hardness andcrispiness offruit andvegetable tissues, orthatwork together with
coagulants tocreate orstrengthen aclot———
Document 7:*Forexample, when neohesperidine DC(which athigh concentrations tastes sweet) isadded toaflavouring which isthen added toafood,
itisable toincrease specific characteristics, such astheperceived
Document 8:Forexample, when neohesperidine DC(which athigh concentrations tastes sweet) isadded toaflavouring which isthen added to
afood, itisable toincrease specific characteristics, such astheperceived
Document 9:——A—food —additive, —which —adds —or—Color, decorative pigment,surface colorant ————————————
Document 10:—acids —Acids —Substances thatincrease theacidity ofafood product orgive itasour taste —————————Acidity regulators —Acidity regulators —
Assistant:Flavor enhancers aresubstances added tofood tosupplement, enhance, ormodify theoriginal taste and/or aroma (Document 5,9,10).They work by
amplifying existing tastes andsmells, increasing theoverall flavor perception, oreven balancing flavor profiles (Document 1).Forexample, monosodium glutamate (E621)
enhances theumami taste infoods likesoups andsauces (Document 1).Unlike imparting acharacteristic taste oftheir own, they intensify what’s already present.
Substances thatmodify sweetness, sourness, orsaltiness canalsobeconsidered flavor enhancers. Compared toother food additives, flavor enhancers differ intheir primary function.
Emulsifiers (Document 9,10)create orstabilize mixtures ofotherwise unmixable substances likeoilandwater, while bulking agents increase thevolume ofafood product without
significantly affecting itscalorie count (Document 10).Essentially, flavor enhancers focus ontaste modification, while emulsifiers address texture andconsistency,
andbulking agents focus onvolume.
Legend:Doc. 1
(-9%)Doc. 2
(-26%)Doc. 3
(-16%)Doc. 4
(-2%)Doc. 5
(-10%)Doc. 6
(-1%)Doc. 7
(+14%)Doc. 8
(+8%)Doc. 9
(-2%)Doc. 10
(+14%)
Figure 13.Visual example of token-level attributions for FSS (FoodSafety) query. Retriever: Snowflake Arctic Embed, Generator: Gemma
3 12B.
20

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Query:How haspopmusic been integrated intomusic education?
Document 1:Throughout itsdevelopment popmusic hasabsorbed influences from most other genres ofpopular music Early popmusic
drew onthesentimental ballad foritsform gained itsuseofvocal harmonies from gospel and
Document 2:PopMusic isbasically agenre ofmusic thathasrhythmic elements melodies andhooks This term wasfirstused intheyear 1926
Itisconsidered asalighter alternative torock androll
Document 3:Popular music pedagogy alternatively called rock music pedagogy modern band popular music education orrock music education
isarecent development inmusic education consisting ofthesystematic teaching andlearning ofrock music and
Document 4:TheNew York Pops isdedicated tolifelong learning andcollaborates with public schools community organizations childrens hospitals
andsenior centers throughout thefiveboroughs ofNew York City PopsEd allows thousands ofNew Yorkers
Document 5:LessonBased Residencies allow professional musicians towork directly with aschools existing band orchestra orchoir through small group
sectionals andor large group rehearsals byfocusing oninstrument technique aswell astheoretical andstylistic
Document 6:Popular music pedagogy alternatively called popular music education rock music pedagogy orrock music education isadevelopment inmusic
education consisting ofthesystematic teaching andlearning ofpopular music both inside andoutside
Document 7:TheNew York Pops isdedicated tolifelong learning andcollaborates with public schools community organizations childrens hospitals
andsenior centers throughout thefiveboroughs ofNew York City PopsEd allows thousands ofNew Yorkers
Document 8:Popmusic issimply short fortheterm popular music Meaning simply thatalarge ofpeople findasong ora
music actenjoyable Overall itwaseveryday people who made itsoavery
Document 9:TheUKhaspioneered theteaching ofpopular music thefirstdegree programme beginning asearly as1994 atSalford University Postgraduate
programmes were later introduced forexample attheInstitute ofPopular Music at
Document 10:PopMusic
Assistant:Popmusic’s integration intomusic education isarelatively recent development, known aspopular music pedagogy (Documents 3&6).
This approach emphasizes group improvisation andoften occurs incommunity settings rather than traditional school ensembles.
TheUKwasapioneer, introducing thefirstdegree program in1994, which subsequently influenced other countries likeGermany,
Ghana, andAustralia (Document 9). In New York City, ”PopsEd,” aprogram runbyTheNew York Pops, actively incorporates pop
music intoeducation (Documents 4&7).Itoffers customizable programs forallages, including composition, lyric writing,
andperformance, reaching thousands ofchildren annually andproviding freelessons andconcert access. Additionally, professional
musicians work directly with school bands andorchestras through ”Lesson-Based Residencies,” focusing ontechnique and
stylistic elements (Document 5).
Legend:Doc. 1
(+6%)Doc. 2
(+3%)Doc. 3
(+10%)Doc. 4
(+13%)Doc. 5
(+14%)Doc. 6
(+15%)Doc. 7
(+20%)Doc. 8
(+1%)Doc. 9
(+13%)Doc. 10
(+5%)
Figure 14.Visual example of token-level attributions for TREC query. Retriever: Snowflake Arctic Embed, Generator: Gemma 3 12B.
21

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Query:Asaconsumer interested infood safety, canyouelaborate ontheessential control measures thatfood establishments should implement tomanage
foodborne illness riskfactors, andhow regulatory authorities assess these measures during inspections?
Document 1:Annex 4oftheFood Code details theessential control measures specific toeach food preparation process, inaddition toessential facility-wide
control measures. Inspectors should generally focus their inspections onverifying thatoperators have
Document 2:Theterm ”routine inspection hasbeen used todescribe periodic inspections conducted aspartofanon-going regulatory scheme.
Program managers should strive tohave adequate staffing andresources toallow allinspectors ample time
Document 3:Since most consumers receive their food from retail andfood service establishments, asignificant share oftheresponsibility forproviding
safefood totheconsumer rests with these facilities. Working together with their regulatory authorities,
Document 4:Inspections, forpurposes ofthiscalculation, include routine inspections, re-inspections, complaint investigations, outbreak investigations, compliance follow-up
inspections, riskassessment reviews, process reviews, variance process reviews, andother direct establishment contact time such ason-site training.
Document 5:On-site corrections made during theinspection should beacknowledged ontheinspection report andintheclosing conference. Thecompliance plan should
address changes inprocedures thatwillprevent therecurrence ofnoted violations. The
Document 6:Thedata collection wasrepeated again in2008 andpublished ina2009 report This wasfollowed byaTrend Analysis Report covering 1998-2008
thatwaspublished inOctober 2010. TheCDC Surveillance Report and
Document 7:ThePresidents Council onFood Safety http://www.foodsafety.gov/˜fsg/cstrpl-4.html#chap2 1While every player intheflow offood from farm totable hassome degree
ofresponsibility forfood safety, youareusually thelastlineof
Document 8:Aspresented inAnnex 4,theCenters forDisease Control andPrevention (CDC) Surveillance Report for1993-1997, Surveillance forFoodborne Disease Outbreaks
United States (http://www.cdc.gov/mmwr/preview/mmwrhtml/ss4901a1.htm )identifies themost frequently reported contributing factors to
Document 9:*Tomonitor theepidemiological evolution offoodborne illnesses anddescribe thecharacteristics offoodborne illnesses *Identify foods, products atrisk
and/or pathogens inorder tostop transmission, orient control andprevention measures and
Document 10:—CHAPTER 3—DEVELOPING YOUR FOOD SAFETY MANAGEMENT SYSTEM .21——Agencies
—59——Articles —62——Books .....—64——FDA Publications and
Assistant:Food establishments areresponsible forpreventing foodborne illnesses through active managerial control, which involves incorporating specific actions and
procedures intotheir operations (Document 1,2,6).This doesn’t necessarily require aformal HACCP plan, though it’srecommended (Document 1).
Instead, establishments should focus onpreventing, eliminating, orreducing foodborne illness riskfactors. These riskfactors, asidentified bytheCDC and
FDA, include food from unsafe sources, inadequate cooking, improper holding temperatures, contaminated equipment, andpoor personal hygiene (Document 3,8).
Tocontrol these, establishments should implement control measures liketraining programs, manager oversight, standard operating procedures, andincorporating
safety intorecipes andjobdescriptions (Document 1).Elements ofaneffective food safety management system may include certified food protection
managers, standard operating procedures, recipe cards, andemployee health policies (Document 6). Regulatory authorities, likehealth inspectors,
assess these control measures during inspections byobserving behaviors, practices, andprocedures, andasking questions (Document 4).
They focus onverifying theimplementation ofFood Code interventions, such asdemonstrating knowledge, implementing employee health policies,
andensuring proper time/temperature control (Document 2).Inspectors alsoevaluate whether managers andemployees understand food safety principles and
Legend:Doc. 1
(+21%)Doc. 2
(+18%)Doc. 3
(-5%)Doc. 4
(+7%)Doc. 5
(+0%)Doc. 6
(+19%)Doc. 7
(+3%)Doc. 8
(+7%)Doc. 9
(-13%)Doc. 10
(+7%)
Figure 15.Visual example of token-level attributions for FSS Low WARG query. Retriever: Snowflake Arctic Embed, Generator: Gemma
3 12B.
22

RAG-E: Quantifying Retriever-Generator Alignment and Failure Modes
Query:What arelung cancer’s symptoms?
Document 1:Themost common symptoms oflung cancer are1Acough thatdoes notgoaway orgetsworse 2Chest pain thatisoften
worse with deep breathing coughing orlaughing 3Hoarseness 4
Document 2:Themost common symptoms oflung cancer are1Acough thatdoes notgoaway orgetsworse 2Chest pain thatisoften
worse with deep breathing coughing orlaughing 3Hoarseness 4
Document 3:Themost common symptoms oflung cancer are1Acough thatdoes notgoaway orgetsworse 2Coughing upblood or
rustcolored sputum spitorphlegm 3Chest pain thatisoften
Document 4:Themost common lung cancer symptoms include 1Appetite loss2Fatigue 3Weight loss4Apersistent cough
5Achange inalong standing cough 6Breathlessness orshortness ofbreath 7Coughing
Document 5:Themost common lung cancer symptoms include 1Appetite loss2Fatigue 3Weight lossApersistent 1cough
Achange inalong standing cough Breathlessness orshortness of1breath Coughing upblood
Document 6:Common symptoms oflung cancer include 1Acough thatdoesnt goaway andgetsworse over time
2Constant chest pain 3Coughing upblood Shortness ofbreath wheezing or1hoarseness Repeated problems
Document 7:Signs andsymptoms oflung cancer may include 1Anew cough thatdoesnt goaway 2Changes inachronic
cough orsmokers cough 3Coughing upblood even asmall amount Shortness of
Document 8:Signs andsymptoms oflung cancer typically occur only when thedisease isadvanced Signs andsymptoms oflung cancer
may include Anew cough thatdoesnt goaway Changes inachronic cough or
Document 9:Signs andsymptoms oflung cancer typically occur only when thedisease isadvanced Signs andsymptoms oflung cancer
may include Anew cough thatdoesnt goaway Coughing upblood even asmall
Document 10:Common symptoms oflung cancer include Acough thatdoesnt goaway andgetsworse over time Constant chest pain
Coughing upblood Shortness ofbreath wheezing orhoarseness Repeated problems with pneumonia orbronchitis
Assistant:Lung cancer symptoms canvary, butseveral arecommonly reported. Apersistent orworsening cough isaprimary indicator
(Documents 1,2,3,4,6,7,8,9,10).Many sources alsomention chest pain, often exacerbated bybreathing, coughing,
orlaughing (Documents 1,2,3,6,8). Other frequent symptoms include hoarseness (Documents 1,2,3,6,7),weight loss,
andlossofappetite (Documents 1,2,3,4,5,6,10).Coughing upblood orrust-colored sputum isaconcerning sign (Documents 2,
4,5,7,10).Shortness ofbreath orbreathlessness isalsocommon (Documents 3,4,5,7,8,10).Finally, fatigue, aches orpains
when breathing/coughing, andrepeated chest infections canoccur (Documents 4,5,6,7,8). **Please note:**
Symptoms often appear when thedisease isadvanced (Documents 8,
Legend:Doc. 1
(+16%)Doc. 2
(+12%)Doc. 3
(+14%)Doc. 4
(+9%)Doc. 5
(+13%)Doc. 6
(+2%)Doc. 7
(+8%)Doc. 8
(+12%)Doc. 9
(+7%)Doc. 10
(+7%)
Figure 16.Visual example of token-level attributions for TREC Low WARG query. Retriever: Snowflake Arctic Embed, Generator:
Gemma 3 12B.
23