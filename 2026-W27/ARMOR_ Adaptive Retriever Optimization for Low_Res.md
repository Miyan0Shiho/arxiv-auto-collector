# ARMOR: Adaptive Retriever Optimization for Low-Resource Telecom Question Answering

**Authors**: Heshan Fernando, Quan Xiao, Yan Xin, Tianyi Chen

**Published**: 2026-06-29 02:18:51

**PDF URL**: [https://arxiv.org/pdf/2606.29706v1](https://arxiv.org/pdf/2606.29706v1)

## Abstract
Telecom question answering (QA) is a challenging setting for retrieval-augmented generation (RAG): evidence is fragmented across standards, papers, encyclopedic resources, and web documents, and answers often hinge on technical tables, equations, and specialized protocol language. In low-resource subdomains, generator fine-tuning can over-specialize and degrade general capability, making query-side retriever adaptation an attractive alternative. To this end, we ask whether a fixed-generator, query-adapted RAG system can outperform generator-side adaptation, and which retriever objectives best support that setting. We motivate retrieval, rather than generator fine-tuning, as the adaptation target through a capacity comparison: under bounded-parameter and soft-retrieval assumptions, query-encoder tuning can have a smaller estimation term than supervised fine-tuning when its effective dimension is smaller. We identify two particularly relevant objectives -- the latent-document RAG likelihood, which optimizes generation utility, and the InfoNCE contrastive objective, which improves semantic retrieval geometry -- and leverage them jointly through a retriever optimization method targeting downstream QA performance in the telecom domain. Specifically, we introduce ARMOR, Adaptive Regularized Mixture Optimization for Retrievers, which learns separate temperatures for the RAG retrieval distribution and InfoNCE softmax and regularizes the adapted query encoder toward the frozen base query encoder. Across telecom-specific retrieval and generative QA benchmarks, we show that ARMOR improves evidence retrieval and answer generation in several in-domain settings. Code is available at https://github.com/heshandevaka/ARMOR.git.

## Full Text


<!-- PDF content starts -->

ARMOR: Adaptive Retriever Optimization for
Low-Resource Telecom Question Answering
Heshan Fernando1Quan Xiao2Yan Xin3Tianyi Chen2, 1
1Rensselaer Polytechnic Institute, Troy, NY
2Cornell University, New York, NY
3Samsung Research America, Berkeley Heights, NJ
fernah@rpi.edu,
{qx232,tianyi.chen}@cornell.edu,
yan.xin@samsung.com
Abstract
Telecom question answering (QA) is a challenging setting for retrieval-augmented
generation (RAG): evidence is fragmented across standards, papers, encyclopedic
resources, and web documents, and answers often hinge on technical tables, equa-
tions, and specialized protocol language. In low-resource subdomains, generator
fine-tuning can over-specialize and degrade general capability, making query-side
retriever adaptation an attractive alternative. To this end, we ask whether a fixed-
generator, query-adapted RAG system can outperform generator-side adaptation,
and which retriever objectives best support that setting. We motivate retrieval, rather
than generator fine-tuning, as the adaptation target through a capacity comparison:
under bounded-parameter and soft-retrieval assumptions, query-encoder tuning
can have a smaller estimation term than supervised fine-tuning when its effective
dimension is smaller. We identify two particularly relevant objectives—the latent-
document RAG likelihood, which optimizes generation utility, and the InfoNCE
contrastive objective, which improves semantic retrieval geometry—and leverage
them jointly through a retriever optimization method targeting downstream QA
performance in the telecom domain. Specifically, we introduce ARMOR, Adaptive
Regularized Mixture Optimization for Retrievers, which learns separate temper-
atures for the RAG retrieval distribution and InfoNCE softmax and regularizes
the adapted query encoder toward the frozen base query encoder. Across telecom-
specific retrieval and generative QA benchmarks, we show that ARMOR improves
evidence retrieval and answer generation in several in-domain settings. Code is
available athttps://github.com/heshandevaka/ARMOR.git.
1 Introduction
Telecom question answering (QA) is challenging for off-the-shelf LLMs because relevant evidence is
fragmented across standards, research papers, encyclopedic resources, and web documents, while
correct answers often depend on equations, tables, protocol logic, and cross-references. Recent
resources such as Tele-Data and Tele-Eval make this setting tractable by pairing telecom corpora
with QA supervision and provenance, naturally favoring grounded retrieval-augmented generation
(RAG) over purely parametric knowledge storage [ 26]. In low-resource subdomains, however, full
generator adaptation is brittle: labeled data are scarce, sequential tuning risks catastrophic forgetting
[21], and prior telecom studies suggest that LoRA fine-tuning may provide limited improvements
when the base model contains only partial domain knowledge [ 14,26]. RAG offers a more modular
alternative: when the generator is already partially capable, improving which evidence it sees can be
more effective than changing the generator itself [22].
Preprint.arXiv:2606.29706v1  [cs.IR]  29 Jun 2026

Base Gen Tele PT Gen Domain SFT Base RAG RAG QE FT RAG LM+QE FT RAFT REPLUG QE FT InfoNCE QE FT 0.00.10.20.30.40.50.60.70.8Average Score0.24920.29720.25990.73770.7604
0.22890.5119 0.51580.7522 No tuning
Generator tuning
Query embedding tuning
Generator + query tuning
RAG enabledFigure 1:Comparison of optimization targets and the performance of retriever-training objec-
tives for ISAC domain QAwith Llama-3-8B-Instruct [ 11] as the generator model and e5-large-v2
[38] as the dense retriever backbone. Retriever-side query-encoder optimization produces substan-
tially larger in-domain gains than either the base generator or generator-side adaptation, motivating
our study of which component to optimize and which retriever objective best improves RAG perfor-
mance. The compared methods areBase Gen, the closed-book generator without retrieval;Tele PT
Gen, a telecom-domain pretrained generator model [ 26];Domain SFT, a domain-specific supervised
fine-tuned generator model;Base RAG, retrieval-augmented generation with the frozen base retriever;
RAG QE FT, query-encoder fine-tuning under the original RAG marginal likelihood objective [ 22];
REPLUG QE FT, query-encoder fine-tuning using the REPLUG objective [ 34];RAFT, query-encoder
fine-tuning using the RAFT objective [ 40];InfoNCE QE FT, query-encoder fine-tuning under the
supervised contrastive InfoNCE objective [17].
We therefore study a simple but consequential question: when domain supervision is scarce and
the document index is fixed, can query-side retriever adaptation beat generator tuning, and which
retrieval objective should drive it? Figure 1 previews the empirical answer on the Integrated Sensing
and Communication (ISAC) domain: query-encoder optimization produces larger in-domain gains
than either closed-book generation or generator-side adaptation. This motivates a retriever-centric
strategy for low-resource telecom QA: keep the generator fixed, adapt only the query interface, and
preserve the frozen document index used at inference time.
However, choosing the retriever as the adaptation target does not by itself determine how to train it.
The latent-document RAG likelihood rewards passages that improve answer generation [ 22], while
InfoNCE improves semantic separation and retrieval discrimination [ 19,17,37]. These signals are
complementary, but they also raise two design challenges. First, a single objective can overemphasize
either downstream utility or embedding geometry, and a static mixture assumes the right balance is
fixed throughout training. Second, because only the query encoder is updated while the document
encoder and index remain frozen, retriever fine-tuning can move query embeddings away from the
base document space. In low-data settings, this drift can improve the training objective without
reliably improving evidence coverage at test time.
We address these issues withARMOR(Adaptive Regularized Mixture Optimization for Retrievers).
ARMOR learns separate temperatures for the RAG retrieval distribution and the InfoNCE softmax,
allowing each objective’s sharpness and influence to change during training, and regularizes the
adapted query encoder toward the frozen base query encoder to preserve compatibility with the fixed
document space. In the main source-document-split Tele-Eval setting, ARMOR improves the ISAC
answer score from 0.6893 to 0.7119 while also achieving the best Recall@3 and Recall@5; across
ISAC, JCC, and SAGIN, it improves evidence coverage at higher recall ranks while avoiding the
answer-quality degradation seen in other fine-tuned retriever baselines.
Contributions.The main contributions are:
C1) Retriever-centric adaptation for low-resource telecom QA.We provide a capacity-based
motivation and empirical evidence that, when domain supervision is scarce, query-encoder
2

tuning can offer a favorable estimation–performance tradeoff relative to generator fine-tuning
while preserving the modularity of a fixed-generator, fixed-index RAG system.
C2) Adaptive objective balancing for retriever optimization.We show that RAG likelihood
and InfoNCE capture complementary retriever signals, and introduce objective-specific
learnable temperatures so their influence can change during training without manually
selecting a fixed mixture weight.
C3) Base-compatible query regularization for telecom QA.By distilling adapted query
embeddings toward the frozen base query encoder, ARMOR limits drift from the fixed
document embedding space. Across telecom retrieval and QA benchmarks, ARMOR gives
the strongest overall tradeoff across in-domain retrieval quality, downstream answer quality,
and robustness across generator scales.
2 What to Optimize in Low-Resource Telecom QA?
2.1 Query-Only RAG Setup and Retriever-Centric Adaptation
LetD={d j}N
j=1be a datastore of chunked telecom passages derived from standards, research
papers, and curated domain web sources. Given a telecom question x, the retriever returns a top- kset
Nk(x) ={d i}k
i=1⊂ D, and a generator produces an answer yconditioned on xand the retrieved
evidence. We adopt a dense retrieval model with score
sη(x, d) =f η(x)⊤g(d),
where fηis a trainable query encoder and gis a document encoder. In the most operationally attractive
variant, gis fixed and the corpus is pre-embedded, so adaptation occurs only on the query side. We
denote the retriever embedding dimension byd E, so thatf η(x), g(d)∈RdE.
This query-only setup is well matched to the low-resource telecom regime. In narrow telecom
slices, the generator is often partially capable but fails when shown incorrect or incomplete evidence.
Updating the retriever is therefore a modular intervention that focuses limited supervision on the
component controlling evidence exposure, while avoiding repeated re-indexing and preserving the
base generator’s general capabilities.
This perspective is supported by the empirical pattern previewed in Figure 1: retriever-side opti-
mization produces stronger in-domain gains than either the base model or generator-side fine-tuning.
More importantly, retriever-centric learning aligns with operational constraints. It allows the system
to update evidence access without repeatedly changing the generator, reuses a frozen document index
when only the query encoder is tuned, and keeps grounding behavior interpretable through retrieved
passages.
We next make this architectural choice more explicit by comparing the low-data generalization
behavior of generator fine-tuning and retriever fine-tuning.
2.2 Low-Data Generalization Motivation
The next result is used as a capacity comparison. It isolates one regime in which query-side tuning
can have a smaller estimation term than generator fine-tuning, namely when the retriever’s effective
dimension is smaller under bounded-parameter and soft-retrieval assumptions.
We compare two families of adaptation,SFTandRAG, on the same population distribution Pover
examples z= (x, y) , where xis the question, and yis the target answer. Given Ni.i.d. training
pairs{zi}N
i=1drawn from the population distribution P, letθ∈ E S⊂RdSbe the trainable generator
parameter for SFT, η∈ E R⊂RdRbe the trainable RAG parameters, and ℓS(z;θ) andℓR(z;η) be
the per-sample SFT and RAG loss, respectively.
The population SFT loss is defined as the expected test loss under the true population distribution P,
while the empirical SFT loss is defined as the empirical training loss on finite sample{z i}N
i=1, i.e.
population loss:L S(θ) =E z∼P[ℓS(z;θ)],empirical loss: bLS,N(θ) =1
NNX
i=1ℓS(zi;θ).
3

Similarly, we can define the population and empirical RAG losses as follows.
population loss:L R(η) =E z∼P[ℓR(z;η)],empirical loss: bLR,N(η) =1
NNX
i=1ℓR(zi;η).
A standard goal of generalization theory is to control the gap between the population loss and the
empirical loss. In our comparison, a smaller upper bound should be interpreted as better estimation
control under the stated assumptions rather than as a direct prediction of downstream performance.
We denote the RAG answer distribution as qη(· |x) =P
d∈Nk(x)pη(d|x)p θ0(· |x, d) . Therefore,
letting ℓNLL(q, x, y) =−logq(y|x) be the negative log-likelihood loss, the SFT and RAG loss can
be rewritten as
ℓS(z;θ) =ℓ NLL(pθ, x, y),andℓ R(z;η) =ℓ NLL(qη, x, y).
where the first optimizes the generator parameter θto predict the answer ygiven the question x, and
the latter optimizes the retrieval parameter ηto select the relevant document d, so that the model can
generate the correct answeryconditioned on the resulting question–document pair(x, d).
Theorem 1(Capacity comparison for SFT and RAG).Assume that the loss function
ℓNLL(q, x, y) is bounded in [0, c] and is ρ-Lipschitz with respect to qin the feasible do-
main, and that the weights for SFT and RAG are bounded by Rfor all t. Also assume that
the tokenized inputs for questions and answers (xn, yn)have full rank and are bounded by
RXafter scaling by the corresponding dimensions dRanddS, for all n∈[N] . Additionally,
assume ∥g(d)∥ ≤B gandqηisρsoft-Lipschitz over sη(x, d) on the feasible domain. Then with
probability at least1−δ, we have for anyθ∈ E Sandη∈ E R,
LS(θ)≤bLS,N(θ) + ˜O(ρC(N, R, R X, dS, dm)) + 3cr
log(2/δ)
2N(1)
LR(η)≤bLR,N(η) + ˜O(ρsoftBgp
dEρC(N, R, R X, dR, dm′)) + 3cr
log(2/δ)
2N(2)
whereC(N, R, R X, d, d m) :=q
P(d,d m)
N3
1 + log
RORV(√
dRX)q
N
P(d,d m)
and
P(d, d m) = (√
dRX)2p
dmRV2
3+p
dmRKRQRV2
33
log (Nd)
withd mandd m′denoting the corresponding model architecture widths for SFT and RAG.
Theorem 1 builds on generalization theory for Transformer-based models [ 29,36] and retriever tuning
[3]. The proof of Theorem 1 is provided in Appendix B.
Remark 1.When the empirical losses are comparable and Nis small, the complexity term
can dominate the comparison between the two upper bounds. Omitting the logarithmic factors,
the SFT complexity term scales as ˜O(ρ√dSdm
N3/2), whereas the RAG complexity term scales
as˜O(ρρsoftBg√
dEdRdm′
N3/2 ). Therefore, when ρsoftBg=O(1) , the RAG upper bound can be
tighter wheneverd EdRdm′≲dSdm.
In our experiments, query-encoder adaptation uses about four times fewer trainable parameters than
generator adaptation, which is consistent with the dimensional condition above. Theorem 1 should
therefore be read as a motivating capacity argument rather than a standalone explanation of empirical
superiority. It helps explain why retriever tuning is a plausible low-data adaptation target, while the
results in Figure 1 and Section 5.2 determine whether this advantage appears in the evaluated telecom
QA setting. The remaining design question is which retriever objectives should drive this adaptation.
4

2.3 Retriever Objectives: RAG Likelihood, InfoNCE, and Static Mixing
Following the empirical pattern in Figure 1, we focus on two complementary retriever objectives:
RAG likelihood and InfoNCE.
RAG likelihood.With a frozen generator pθ0(y|x, d) , the RAG objective optimizes retrieval
indirectly through answer likelihood:
LRAG(x, y) =−logX
d∈Nk(x)pη(d|x)p θ0(y|x, d),
whereN k(x)is the top-kdocument set ranked by the retrieval scores η(x, d)and
pη(d|x) =exp (s η(x, d))P
d′∈Nk(x)exp (s η(x, d′)).
This objective rewards passages that are useful for generation, not merely passages that are topically
related.
InfoNCE.Given a positive passaged+and a set of negativesN−, the contrastive objective is
LInfoNCE (x, d+,N−) =−logexp(s η(x, d+))
exp(s η(x, d+)) +P
d−∈N−exp(s η(x, d−)).
In telecom QA, positive passages can often be obtained from source-linked datasets such as Tele-Eval,
while negatives may come from in-batch sampling, BM25, or hard-negative mining. This objective
primarily improves semantic separability and retrieval discrimination.
These objectives matter for different reasons. RAG likelihood is utility-driven: it improves the
retriever to the extent that retrieval helps the generator answer correctly. InfoNCE is geometry-driven:
it improves the embedding space so that relevant evidence is easier to recover. In low-resource
telecom QA, both signals are needed. RAG alone may under-shape the representation space, while
InfoNCE alone may reward topical similarity without fully aligning retrieval with downstream answer
utility. Empirically, both objectives perform well in Figure 1, which motivates combining them rather
than treating either objective as sufficient on its own.
A standard mixture can be written as
Lstatic =λL RAG+ (1−λ)L InfoNCE ,
where λ∈[0,1] is fixed in advance. The static mixture has two limitations that ARMOR targets
directly: it fixes the objective balance for all training stages, and it does not constrain the adapted
query encoder to remain compatible with the frozen document index.
3 ARMOR: Adaptive and Regularized Retriever Optimization
3.1 Temperature-Modulated Retriever Objectives
The fixed mixture in Lstatic assumes that the relative influence of RAG and InfoNCE should remain
constant throughout training. This is a strong assumption: the two objectives capture different aspects
of retriever learning, and their relative usefulness need not be the same early and late in optimization.
Kendall et al. [ 20] make a related point in multi-task learning, showing that fixed loss weighting can
bias training toward one objective and that adaptive weighting can lead to more balanced optimization.
Our setting differs in that we optimize a single retriever with two complementary signals, and we
place adaptivity inside the softmax distributions that define the objectives rather than on external loss
coefficients. We therefore introduce learnable positive temperatures into both retriever objectives. In
implementation, we optimize unconstrained scalarsα randα cand set
τr=τmin+ softplus(α r), τ c=τmin+ softplus(α c),
where τmin>0is a small floor used only to avoid degenerate zero-temperature softmaxes. For RAG,
the retriever distribution over the top-kdocuments becomes
p(τr)
η(di|x) =exp(s η(x, d i)/τr)
Pk
j=1exp(s η(x, d j)/τr),
5

which yields the temperature-modulated loss
LRAG(x, y;τ r) =−logkX
i=1p(τr)
η(di|x)p θ0(y|x, d i).
For InfoNCE, we similarly define
LInfoNCE (x, d+,N−;τc) =−logexp(s η(x, d+)/τc)
exp(s η(x, d+)/τc) +P
d−∈N−exp(s η(x, d−)/τc).
Large temperatures produce smoother distributions and broader supervision; small temperatures
sharpen the objectives and focus training on top-ranked or hardest competing items. We then define
the adaptive retriever objective as
Lmix(η, α r, αc) =L RAG(x, y;τ r) +L InfoNCE (x, d+,N−;τc).
3.2 Gradient Interpretation of Adaptive Temperature
Adaptive temperatures can be viewed as objective-side parameters that control how strongly the RAG
and InfoNCE losses shape the query encoder. This explains how temperature learning acts as an
implicit adaptive weighting mechanism, even though ARMOR does not attach explicit scalar mixture
weights to the RAG and InfoNCE losses themselves.
We first consider the contrastive objective in a minimal two-document setting with one positive
document d+, one negative document d−, and query embedding q. Lets+=q⊤d+ands−=q⊤d−,
and define the margin∆ =s+−s−. The InfoNCE loss with temperatureτ cis
LNCE(q, τc) = log(1 + exp(−∆/τ c)).
Differentiating with respect to the query embedding yields
∇qLNCE=−1
τcσ
−∆
τc
(d+−d−),
where σ(·) is the logistic sigmoid. Thus, τcdirectly scales the gradient through 1/τcand controls
how sharply updates focus on hard-margin examples. Smaller τcmakes the contrastive loss act more
aggressively on the query encoder, while largerτ csmooths its influence.
An analogous effect appears in the retrieval component of the RAG objective. Consider a top- 2
setting with score gap ∆ =s 1−s2between two retrieved documents. The temperature-modulated
retrieval distribution isp 1=σ(∆/τ r)andp 2= 1−p 1, giving the RAG loss
LRAG(q, τr) =−log 
p1a1+p2a2
,
where a1anda2denote the generator-side answer utilities of the two retrieved documents. Differ-
entiating with respect to the score gap again yields an explicit 1/τrfactor. Hence, smaller retrieval
temperature makes the loss more sensitive to differences among top-ranked documents, while larger
temperature spreads the training signal more broadly.
Together, these observations show that temperatures do more than change softmax entropy: they
determine how strongly each objective contributes gradient signal to the query encoder. In this
sense,τ randτ cact as learned, objective-specific weighting mechanisms. Decreasing a temperature
sharpens the corresponding objective and increases its local influence, while keeping it large makes
the objective more conservative. This view explains how adaptive temperatures can replace manually
tuned mixture weights while still allowing the relative influence of RAG likelihood and InfoNCE to
evolve during training.
The main consequence of this formulation is that adaptive temperature learning provides a mechanism
forbalanced retriever optimization. Rather than fixing the sharpness of the two objectives in advance,
the model can begin training in a smoother regime and progressively sharpen one or both objectives as
the retriever becomes more reliable. This makes the approach particularly appealing in low-resource
settings, where fixed weighting can otherwise be brittle and highly sensitive to tuning.
At the same time, the same mechanism also introduces a potential failure mode. Once the retriever
is already locally correct on many training examples, gradient descent can continue driving the
6

0 500 1000 1500 2000
Training Step246810Temperature
ISAC
0 500 1000 1500 2000
Training Step
JCC
0 500 1000 1500 2000
Training Step
SAGIN
0.010.020.03
Regularization
r
 c
LqdistFigure 2:Training dynamics of adaptive temperatures and query-distillation regularization
across ISAC, JCC, and SAGIN domains.Retrieval temperature consistently sharpens during
training, while query-distillation loss rises late, indicating increasing tension between domain special-
ization and compatibility with the frozen embedding space.
temperatures downward, eventually making the softmaxes excessively sharp. In that regime, tem-
perature learning no longer improves balance, but instead acts as a shortcut that amplifies existing
score differences. This observation motivates the regularization strategy introduced below, which
constrains the query encoder to remain compatible with the frozen base document space even as the
retriever objectives become sharper.
3.3 Query Distillation and Final ARMOR Objective
Only the query encoder is updated in our setting, while the document encoder and index remain fixed
in the base embedding space. As a result, improving the mixed retriever objective alone can still
move the query encoder away from the geometry used at inference time. This issue is especially
acute in low-resource domains, where optimization can over-specialize to the training objective.
To preserve compatibility with the frozen document space, we regularize the adapted query encoder
toward the frozen base query encoder. For a minibatch of queries, we use cosine query distillation,
Lqdist=1
BBX
b=1 
1−cos 
qη(xb), q0(xb)
,
where qη(xb)is the adapted query embedding and q0(xb)is the corresponding frozen base embedding.
Because both query and document embeddings are ℓ2-normalized, cosine similarity provides a natural
measure of deviation from the original retrieval geometry.
The adaptive temperatures and the query-distillation regularizer together define our full training
framework,ARMOR(AdaptiveRegularizedMixtureOptimization forRetrievers). ARMOR
combines adaptive mixture optimization of the RAG and InfoNCE objectives with regularization that
preserves compatibility with the frozen base retriever space. The resulting objective is
LARMOR (η, α r, αc) =L RAG(x, y;τ r) +L InfoNCE (x, d+,N−;τc) +λ qLqdist
where λq≥0controls the strength of query distillation. We choose λq= 1for ARMOR; setting
λq= 0gives the dynamic-temperature variant without query regularization used in the ablation. Thus,
the adaptive temperatures determine how selectively the utility-driven and geometry-driven objectives
shape the query encoder, while λqcontrols the compatibility penalty that prevents excessive drift
away from the pretrained retrieval space.
4 Related Work
Retrieval-Augmented Architectures.Retrieval-augmented systems use external evidence as non-
parametric memory to complement model weights. REALM and RAG connected retrieval to
downstream learning by treating documents as latent variables [ 12,22], while FiD, Atlas, RETRO,
and in-context retrieval-augmented language models established the benefits of modular memory,
multi-passage reasoning, and updatable indices [ 16,18,4,2]. REPLUG is closest to our framing
because it keeps the language model fixed and adapts prediction through retrieval [ 34]. However,
7

0 250 500 750 1000 1250
Number of datapoints0.620.640.660.680.700.720.74Average score
Base RAG
ARMOR0.66160.6821
0.65940.67990.6897
0.67710.7126
0.65590.67470.7023
0.68030.70910.7214
0.7119(a) Training examples
0 200 400 600 800 1000 1200
Number of documents
0.69030.70070.6967
0.68550.7023
0.68270.71430.7303
0.70320.7069
0.69930.7119 (b) Source documents
arxiv standard wiki
Document type0.7140
0.67710.6861 (c) Document type
Figure 3:ISAC Tele-Eval data and corpus ablations.(a) and (b) vary ARMOR training supervision
and source-document coverage; (c) restricts training to one source category. Performance improves
overall but remains sensitive to source alignment and corpus composition.
REPLUG-style prediction can incur additional test-time overhead from document-wise scoring or
aggregation, whereas our method retains the standard RAG-style retrieval-and-generation interface of
Lewis et al. [22] while modifying the retriever-training objective to improve downstream performance.
Dense Retrieval and Contrastive Learning.Dense passage retrieval introduced the dual-encoder
paradigm for open-domain QA, embedding queries and documents in a shared space with contrastive
supervision [ 19]. Contriever shows that such objectives can produce strong retrievers with limited
labeled data, making them well suited to low-resource adaptation [ 17]. InfoNCE formalizes this
discrimination-based training by contrasting positives and negatives, with temperature controlling
softmax sharpness [ 37]. In contrast to purely contrastive retriever adaptation, we combine contrastive
learning with generation-focused retrieval objectives and use adaptive temperature control to balance
their influence during retriever fine-tuning.
Telecom Grounding and Evaluation.Tele-Data and Tele-Eval provide telecom-specific corpora, QA
supervision, and source identifiers for grounded QA [ 26]. KILT highlights the need to evaluate both
answer correctness and evidence recovery in knowledge-intensive tasks [ 30], while RAGAS offers
reference-free measures of faithfulness and context relevance when exact citations are incomplete
or noisy [ 7]. This is critical in telecom, where plausible but unsupported answers are insufficient.
Building on these resources, we identify which components of a telecom RAG system should be
adapted for domain-specific knowledge while limiting overfitting in low-resource settings.
Regularization and Objective Balancing.Balancing multiple training signals is a recurring problem
in multi-task and multi-objective learning. One line of work reweights task losses using uncer-
tainty [ 20], gradient magnitudes [ 5], or multi-objective formulations [ 32]. Another line works directly
with gradients, aggregating or correcting them to reduce conflict and stochastic bias [ 39,24,8,9].
Penalty-based reformulations provide a related way to handle constrained or hierarchical optimization
structure [33].
Similar trade-offs also appear in LLM post-training. Sequential SFT and preference learning can
degrade safety, alignment, or earlier task performance [ 31,23], motivating methods that reformulate
preference optimization, unify supervised and reinforcement-style objectives, or analyze and improve
supervised–preference trade-offs [13, 15, 6, 25, 10].
ARMOR is closest in spirit to adaptive loss weighting, but the adaptation is placed inside the retriever
objectives rather than on external task coefficients. It learns temperatures for the RAG likelihood
and InfoNCE softmax, then pairs this objective balancing with query distillation. This pairing is
specific to the query-only RAG setting: because the document encoder and index remain frozen,
regularization is needed to prevent over-sharpened retrieval and contrastive signals from moving the
query encoder away from the fixed document embedding space.
5 Experiments
We evaluate ARMOR on domain-specific telecom QA to understand when query-side retriever
adaptation improves both evidence retrieval and downstream answer generation. The experiments
first define a source-document-held-out Tele-Eval setting for in-domain evaluation, then examine
main QA and retrieval performance, component ablations, ARMOR training dynamics, robustness
8

Table 1:Tele-Eval experimental data summary.Documents are matched to Tele-Data using
Tele-Eval source identifiers and chunked into 384-character passages.
Statistic ISAC JCC SAGIN
Tele-Eval QA pairs after filtering 1,701 3,060 1,339
Unique source documents matched 1,406 2,486 1,000
Indexed chunks 105,476 115,122 67,704
Final train size 1,361 2,466 1,048
Final validation size 178 294 137
Final test size 162 300 154
across training data size and corpus composition, out-of-corpus TeleQnA evaluation, and generator
scale.
5.1 Experimental Setup
Data and splits.The main experiments focus on three representative 6G subdomains: Integrated
Sensing and Communication (ISAC), Joint Communication and Computation (JCC), and Space-
Air-Ground Integrated Networks (SAGIN). We use two public resources: Tele-Eval [ 26], a domain-
grounded telecom QA dataset, and Tele-Data [ 26], a large-scale telecom corpus containing standards,
arXiv, and Wikipedia documents. For each retained Tele-Eval QA pair, the source document identifier
is matched back to Tele-Data to construct a domain-specific retrieval corpus. Documents are split
into 384-character passages and indexed once with the baseintfloat/e5-large-v2encoder.
To reduce leakage, we split at the source-document level. Ten percent of unique source documents
are reserved exclusively for testing; no QA pair derived from those documents appears in training or
validation. Candidate chunks from each matched source document are then scored to identify the
top three positive passages per question. These aligned positives provide contrastive supervision and
define the ground-truth targets for Recall@k. Table 1 summarizes the resulting data.
Models and baselines.The dense retriever backbone is intfloat/e5-large-v2 . Document
embeddings are fixed in the base embedding space throughout training; only the query encoder is
updated. The primary generator is Llama-3-8B-Instruct, with additional generator-scale comparisons
using Llama-3.2-1B, Llama-3.2-3B [ 11], and Qwen3-8B [ 35]. We compareBase Gen(closed-book
generation),Base RAG(frozen retriever),RAG QE FT(query-encoder fine-tuning with RAG
likelihood),InfoNCE QE FT(supervised contrastive fine-tuning),Mix QE FT(static mixture of
RAG likelihood and InfoNCE), andARMOR(adaptive temperatures plus query distillation).
Evaluation.Tele-Eval measures open-ended QA and retrieval fidelity on the source-document-held-
out test split. Unless otherwise noted, all reported Tele-Eval metrics use the first 150 examples
from each domain’s held-out test split. The retriever returns top-16 chunks, which are prepended
to the prompt; the generator then produces a free-form answer. A GPT-5.2 judge scores the answer
against the gold reference on a 0–1scale, and retrieval quality is measured by Recall@1, Recall@3,
and Recall@5 against the aligned positive chunks. Appendix A.5 lists the LLM judge prompts
and key decoding parameters used in the experiments. TeleQnA [ 27] provides an out-of-corpus
multiple-choice robustness check. We report TeleQnA accuracy at top- k= 16 , matching the retrieval
breadth used for Tele-Eval generation.
5.2 Experiment Results
Main Tele-Eval Results.Table 2 shows that the harder source-document-split setting changes the
empirical picture. Every non-ARMOR adaptation method degrades below Base RAG on at least one
domain’s answer score, whereas ARMOR is the only fine-tuned method that consistently matches or
improves the frozen baseline. On ISAC, ARMOR improves the answer score from 0.6893 to 0.7119
while also achieving the best Recall@3 and Recall@5. On SAGIN, ARMOR gives the strongest
answer score and improves slightly over Base RAG. On JCC, where Base RAG is already strongest,
ARMOR approximately preserves answer quality while giving the best retrieval recall at all reported
ranks.
9

Table 2:Tele-Eval open-ended QA and retrieval results across ISAC, JCC, and SAGIN domains.
Score is the GPT-5.2 judge average answer score; R@ kis recall against aligned positive passages.
Best values within each domain and metric are bolded.
MethodTele-Eval, top-16 retrieval
ISAC JCC SAGIN
Score R@1 R@3 R@5 Score R@1 R@3 R@5 Score R@1 R@3 R@5
Base Gen 0.2269 – – – 0.2980 – – – 0.3017 – – –
Base RAG 0.68930.54670.7400 0.80670.77630.4800 0.6133 0.7000 0.76600.64000.8133 0.8400
RAG QE FT 0.6584 0.5000 0.6533 0.7400 0.7230 0.3867 0.6000 0.6533 0.7573 0.5000 0.6933 0.7867
InfoNCE QE FT 0.6685 0.5200 0.6733 0.7533 0.7425 0.4133 0.6200 0.6800 0.7662 0.5333 0.7400 0.8067
Mix QE FT 0.6854 0.4733 0.6333 0.7133 0.7360 0.4467 0.6400 0.6800 0.7591 0.5000 0.7200 0.7800
ARMOR0.71190.52670.7667 0.82000.77190.4933 0.6467 0.7133 0.76850.62670.8400 0.8467
Table 3:ARMOR component ablation with ISAC Tele-Eval.Adaptive temperatures and query-
distillation regularization are complementary: temperature learning without regularization performs
worst, while full ARMOR performs best.
Method Adaptive Temps. Regularization Avg. Score
Base RAG – – 0.6893
RAG QE FT – – 0.6584
InfoNCE QE FT – – 0.6685
Mix QE FT no no 0.6854
Static Mix with Reg. no yes 0.6729
Dynamic Mix without Reg. yes no 0.6350
ARMOR yes yes0.7119
The retrieval pattern is especially informative: ARMOR dominates R@3 and R@5 in every domain,
even where Base RAG keeps the best R@1. In the top-16 generation setting, downstream answer
quality depends on whether useful evidence appears in the context window, not only on whether the
single highest-ranked passage is positive. ARMOR therefore appears to improve evidence coverage
without collapsing retrieval mass onto a brittle top-1 decision.
ARMOR Component Ablation.Table 3 isolates the contribution of ARMOR’s two design choices.
Dynamic Mix without Reg. performs worse than all other fine-tuned methods, which supports the
central failure mode: unconstrained temperature learning can over-sharpen retrieval distributions and
pull query embeddings away from the frozen document space. Regularization alone does not recover
the full gain either, since Static Mix with Reg. remains below the unregularized static mixture. Only
the combination of adaptive temperatures and query distillation improves substantially over both
single-objective and static-mixture baselines.
Data and Corpus Ablations.Figure 3 studies how ARMOR’s performance changes as the amount
of ISAC training supervision, source-document coverage, and document-type composition vary.
Panels (a) and (b) trend upward overall, but neither is perfectly monotonic. This is expected in a
source-document-split setting: adding examples or documents changes the balance among RAG
likelihood, InfoNCE, and query distillation, and intermediate subsets can be less aligned with the
held-out evaluation set than smaller or larger subsets. Panel (c) provides a complementary view by
restricting training to one document category, reinforcing that retriever adaptation depends not only
on how much data is available, but also on whether the document type matches the evidence needed
at evaluation time.
Training Dynamics.Figure 2 shows that the learned temperatures evolve on different timescales.
Retrieval-side sharpening is robust across domains, while contrastive-side sharpening is more domain-
dependent. The late rise in query-distillation loss is consistent with the ablation in Table 3: adaptive
temperatures are useful because they allow the influence of the two objectives to change during
training, but they require regularization to prevent harmful drift from the frozen document index.
Out-of-Corpus Transfer via TeleQnA.TeleQnA plays a different role from Tele-Eval. Because
it is multiple-choice, the model often needs only enough topical signal to eliminate wrong options,
10

Llama-3.2-1B Llama-3.2-3B Llama-3-8B Qwen3-8B0.00.10.20.30.40.50.60.70.80.9Average Score
0.10180.15870.22690.29980.48910.71000.68930.7058
0.48700.69070.71190.7273Base Gen Base RAG ARMORFigure 4:Comparison of Base Gen, Base RAG, and ARMOR across generator backbones on
ISAC Tele-Eval.ARMOR’s gains are clearest for 8B-scale generators, suggesting that stronger
generators are better able to use improved retrieved evidence.
Table 4:TeleQnA multiple-choice accuracy.TeleQnA is used as an out-of-corpus robustness check
rather than the primary retrieval-sensitive benchmark.
Method ISAC JCC SAGIN
Base Gen 0.7000 0.6000 0.6800
Base RAG 0.82000.76000.7600
RAG QE FT 0.8400 0.7200 0.7800
InfoNCE QE FT 0.8400 0.72000.8200
Mix QE FT0.88000.74000.8200
ARMOR 0.8600 0.74000.8200
so fine-grained retrieval differences can collapse into similar accuracy values. Table 4 shows that
ARMOR remains competitive at top- k= 16 without catastrophic out-of-corpus degradation: it is
within two points of the best result on ISAC and JCC and ties the best result on SAGIN.
Effect of Generator Scale.Figure 4 shows that ARMOR’s benefit over Base RAG grows with
generator capacity rather than shrinking. For Llama-3.2-1B, ARMOR and Base RAG are essentially
tied; for Llama-3.2-3B, ARMOR is slightly below Base RAG; for both 8B models, ARMOR
outperforms Base RAG by roughly two to three points. This suggests that retriever optimization
is most valuable when the generator is capable enough to synthesize and use a better evidence set.
Qwen3-8B also outperforms Llama-3-8B under both Base RAG and ARMOR, indicating that model
family matters in addition to parameter count.
6 Conclusion
We studied low-resource telecom QA and showed that retriever-side adaptation can be a more
effective and operationally stable alternative to generator fine-tuning. We proposedARMOR, which
combines RAG likelihood and InfoNCE through adaptive temperature-based objective shaping,
while using query distillation to preserve compatibility with the frozen document embedding space.
Empirically, ARMOR provides a robust tradeoff between retrieval specialization and downstream
QA performance, with training dynamics showing that adaptive temperatures sharpen the retrieval
signal while regularization limits query-encoder drift. These results suggest that low-resource RAG
adaptation should jointly optimize complementary retriever objectives under compatibility constraints.
References
[1]Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of op-
timization and generalization for overparameterized two-layer neural networks. InInternational
11

conference on machine learning, pages 322–332, 2019.
[2] Akari Asai et al. In-context retrieval-augmented language models.TACL, 2023.
[3]Soumya Basu, Ankit Singh Rawat, and Manzil Zaheer. A statistical framework for data-
dependent retrieval-augmented models. InInternational Conference on Machine Learning,
pages 3197–3223, 2024.
[4]Sebastian Borgeaud et al. Improving language models by retrieving from trillions of tokens.
ICML, 2022.
[5]Zhao Chen et al. Gradnorm: Gradient normalization for adaptive loss balancing in deep
multitask networks. InICML, 2018.
[6]Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans,
Quoc V . Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of
foundation model post-training.arXiv preprint arXiv:2501.17161, 2025.
[7]Shahul Es et al. Ragas: Automated evaluation of retrieval-augmented generation.arXiv preprint
arXiv:2309.15217, 2024.
[8]Heshan Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi
Chen. Mitigating gradient bias in multi-objective learning: A provably convergent stochastic
approach. InInternational Conference on Learning Representations, 2023.
[9]Heshan Fernando, Lisha Chen, Songtao Lu, Pin-Yu Chen, Miao Liu, Subhajit Chaudhury,
Keerthiram Murugesan, Gaowen Liu, Meng Wang, and Tianyi Chen. Variance reduction
can improve trade-off in multi-objective learning. InICASSP 2024–2024 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6975–6979. IEEE,
2024.
[10] Heshan Fernando, Han Shen, Parikshit Ram, Yi Zhou, Horst Samulowitz, Nathalie Baracaldo,
and Tianyi Chen. Understanding forgetting in llm supervised fine-tuning and preference
learning–a convex optimization perspective.arXiv preprint arXiv:2410.15483, 2024.
[11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama
3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm:
Retrieval-augmented language model pre-training. InICML, 2020.
[13] Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without
reference model.arXiv preprint arXiv:2403.07691, 2024.
[14] Edward Hu et al. Lora: Low-rank adaptation of large language models.arXiv preprint
arXiv:2106.09685, 2021.
[15] Ermo Hua, Biqing Qi, Kaiyan Zhang, Yue Yu, Ning Ding, Xingtai Lv, Kai Tian, and Bowen
Zhou. Intuitive fine-tuning: Towards unifying sft and rlhf into a single process.arXiv preprint
arXiv:2405.11870, 2024.
[16] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for
open domain question answering.arXiv preprint arXiv:2007.01282, 2021.
[17] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning.
arXiv preprint arXiv:2112.09118, 2021.
[18] Gautier Izacard et al. Atlas: Few-shot learning with retrieval augmented language models.
arXiv preprint arXiv:2208.03299, 2022.
[19] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. InEMNLP, 2020.
12

[20] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh
losses for scene geometry and semantics. InCVPR, 2018.
[21] James Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks.PNAS, 2017.
[22] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.arXiv preprint arXiv:2005.11401, 2020.
[23] Yong Lin, Lu Tan, Hangyu Lin, Zeming Zheng, Renjie Pi, Han Zhao, Yuan Yao, et al. Speciality
vs generality: An empirical study on catastrophic forgetting in fine-tuning foundation models.
arXiv preprint arXiv:2309.06256, 2023.
[24] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent
for multi-task learning.Advances in Neural Information Processing Systems, 34:18878–18890,
2021.
[25] Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. Uft: Unifying supervised and reinforce-
ment fine-tuning.arXiv preprint arXiv:2505.16984, 2025.
[26] Ali Maatouk, Kenny Chirino Ampudia, Rex Ying, and Leandros Tassiulas. Tele-llms: A series
of specialized large language models for telecommunications.arXiv preprint arXiv:2409.05314,
2024.
[27] Ali Maatouk, Fadhel Ayed, Nicola Piovesan, Antonio De Domenico, Merouane Debbah, and Zhi-
Quan Luo. Teleqna: A benchmark dataset to assess large language models telecommunications
knowledge.IEEE Network, 2025.
[28] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar.Foundations of machine learning.
MIT press, 2018.
[29] Brian Mwigo and Anirban Dasgupta. Generalization bound for a shallow transformer trained
using gradient descent.Transactions on Machine Learning Research, 2026. URL https:
//openreview.net/forum?id=t3iUeMOT8Z.
[30] Fabio Petroni et al. Kilt: A benchmark for knowledge-intensive language tasks. InNAACL,
2021.
[31] Xiangyu Qi, Yi Zeng, Tinghao Xie, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning
aligned language models compromises safety, even when users do not intend to!arXiv preprint
arXiv:2310.03693, 2023.
[32] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In
NeurIPS, 2018.
[33] Han Shen, Quan Xiao, and Tianyi Chen. On penalty-based bilevel gradient descent method.
Mathematical Programming, 214(1–2):539–589, 2025. doi: 10.1007/s10107-025-02194-4.
[34] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke
Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In
Proceedings of the 2024 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pages 8371–8384, 2024. doi:
10.18653/v1/2024.naacl-long.463.
[35] Qwen Team. Qwen3 technical report, 2025. URLhttps://arxiv.org/abs/2505.09388.
[36] Jacob Trauger and Ambuj Tewari. Sequence length independent norm-based generalization
bounds for transformers. InInternational Conference on Artificial Intelligence and Statistics,
pages 1405–1413, 2024.
[37] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive
predictive coding.arXiv preprint arXiv:1807.03748, 2018.
13

[38] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan
Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.
arXiv preprint arXiv:2212.03533, 2022.
[39] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea
Finn. Gradient surgery for multi-task learning. InAdvances in Neural Information Processing
Systems, 2020.
[40] Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E. Gonzalez. Raft: Adapting language models to domain-specific rag.arXiv preprint
arXiv:2403.10131, 2024.
14

A Experiment Details
In this section, we provide additional details on the data generation, baseline implementation, and
experiment setup used in this paper. All experiments were conducted on a server equipped with 8
NVIDIA A100 80GB SXM GPUs.
A.1 Data Sources and Experiment Pipelines
This paper uses two related but distinct data pipelines. The introductory comparison in Figure 1
is a motivation experiment based on an earlier standards-only data-generation pipeline. The main
experiments in Section 5.1 use the newer Tele-Eval source-document-split setup summarized in
Table 1. We separate them here to avoid conflating the motivation figure with the final evaluation
protocol.
Introductory motivation pipeline.For Figure 1, we curated domain-specific training data from
the standards portion of Tele-Data [ 26]. Documents were filtered by domain-specific keywords and
LLM refinement, then chunked with the intfloat/e5-large-v2 tokenizer. Grounded QA pairs
were generated from the retained chunks using an instruction-tuned generator, and positive retrieval
targets were aligned back to source chunks using document identifiers and lexical-overlap signals.
This pipeline was used only to motivate the choice of optimizing the query-side retriever rather than
the generator.
Main experimental pipeline.The main experiments use Tele-Eval QA pairs and match each retained
question to its source document in Tele-Data. The retained document pool spans standards, arXiv, and
Wikipedia sources. After domain filtering, each matched source document is split into 384-character
passages and indexed with FAISS using base intfloat/e5-large-v2 embeddings. The document
index is built once and remains fixed during all retriever training.
Source-document split.To prevent leakage, the main experiments hold out 10% of unique source
document IDs for testing. All QA pairs derived from those documents are assigned to the test split,
while the remaining QA pairs are divided into training and validation sets. This policy evaluates
whether the adapted query encoder generalizes to unseen technical documents rather than merely
memorizing passages from documents seen during training.
Positive passage alignment.For each retained QA pair, candidate chunks from the matched source
document are scored to select the top three positive passages. These positives are used as supervised
targets for InfoNCE training and as the ground-truth evidence set for Recall@1, Recall@3, and
Recall@5. InfoNCE negatives are drawn from retrieved chunks outside the positive set together with
in-batch negatives.
A.2 Baseline Retriever Training Implementations
All retriever-training methods initialize from intfloat/e5-large-v2 ; the document encoder and
document index are fixed, and only the query encoder is updated. The baselines are designed to isolate
which retriever objective and regularization choices are responsible for downstream QA behavior.
RAG QE FT.This baseline optimizes the query encoder with the RAG negative log-likelihood
objective under a frozen generator:
LRAG=−logX
d∈Dpη(d|x)p θ0(y|x, d).(3)
The retriever is trained to assign higher probability to chunks that improve likelihood of the target
answer.
InfoNCE QE FT.This baseline uses supervised contrastive training with the aligned positive chunks
and dynamically selected negatives. It directly improves separation between query embeddings and
aligned evidence chunks.
Mix QE FT.This baseline uses a fixed-weight mixture of the RAG and InfoNCE losses:
Lmix=λ RAGLRAG+λcontLcont.(4)
It tests whether simply combining the two objectives is sufficient without adaptive temperatures or
query regularization.
15

ARMOR variants.The component ablation in Table 3 separates the effects of adaptive temperatures
and query-distillation regularization. Static Mix with Reg. adds query distillation to the static
mixture while keeping fixed objective weights. Dynamic Mix without Reg. learns the unconstrained
temperature parameters αrandαcbut sets λq= 0, thereby removing query distillation. Full ARMOR
learns the same temperature parameters and usesλ q= 1for query distillation.
A.3 Models
For the main experiments, the primary generator is Llama-3-8B-Instruct. Generator-scale analysis
additionally evaluates Llama-3.2-1B, Llama-3.2-3B [ 11], and Qwen3-8B [ 35]. Across all runs, the
dense retriever backbone is intfloat/e5-large-v2 . Document embeddings are always computed
with the frozen base encoder. Query embeddings are initialized from the same encoder and then
adapted by the corresponding retriever-training objective.
A.4 Evaluation Harness and Methodology
Tele-Eval.Tele-Eval is the primary benchmark because it measures open-ended answer generation
and evidence retrieval. For each method, the retriever returns top-16 chunks from the fixed FAISS
index. The generator conditions on these chunks and produces a free-form answer. GPT-5.2 then
grades the generated answer against the gold reference on a continuous [0,1] scale. We report
the mean score over the first 150 test examples in each domain. Retrieval quality is measured by
Recall@1, Recall@3, and Recall@5 against the aligned positive chunks, again on the first 150 test
examples.
TeleQnA.TeleQnA [ 27] is used as an out-of-corpus multiple-choice robustness check. Domain-
specific TeleQnA subsets are constructed by matching questions to ISAC, JCC, and SAGIN profiles.
In RAG settings, the retriever supplies top-16 chunks, and the generator selects one answer option.
Accuracy is the fraction of examples for which the parsed option matches the corrected gold option.
Interpretation.Tele-Eval is more sensitive to retrieval quality because the model must synthesize a
grounded free-form answer from the retrieved evidence. TeleQnA is less retrieval-sensitive because
a topically relevant context can often be enough to eliminate incorrect choices. We therefore treat
Tele-Eval as the main benchmark and TeleQnA as a transfer and robustness check.
A.5 LLM Judge Prompts and Parameters
We used LLM judges in four parts of the data and evaluation pipeline. The prompts below are grouped
in the order in which the corresponding procedures are discussed in the paper. Unless otherwise
stated, OpenAI-based judges used temperature 0.0, parsed the first valid numeric score from the
model output, and clamped numeric scores to[0,1].
Prompts used for the Tele-Data standards data-generation pipeline.These prompts were used
in the earlier standards-only Tele-Data pipeline that supports the introductory motivation experiment.
They are separate from the main Tele-Eval source-document-split experiments.
OpenAI Document Relevance Post-Filter.This judge post-filters cleaned Tele-Data standards
documents before example generation. The default OpenAI model was GPT-5.2, the temperature was
0.0, the relevance threshold was 0.50, the snippet length was 2000 characters, the timeout was 120
seconds, and the maximum number of retries was 6.
OpenAI Document Relevance Post-Filter: system prompt
You are a wireless communications and sensing domain expert curating a tutorial dataset.
16

OpenAI Document Relevance Post-Filter: user prompt
Decide whether this Tele-Data sample is suitable for a tutorial dataset focused on {
domain_name}.
{domain_specific_instructions}
Return ONLY a single number between 0.0 and 1.0 (no JSON, no words):
- 1.0 = clear true positive for a {domain_name} tutorial dataset
- 0.0 = clear false positive
SAMPLE_ID: {sid}
CATEGORY: {cat}
META_HINT: {meta_hint}
CONTENT_SNIPPET:
<<<
{snippet}
>>>
For the ISAC data-generation run, the domain-specific instructions asked the judge to include explana-
tory or instructional content on ISAC/JCAS fundamentals, joint waveform and signal design, sensing
tasks in communication systems, joint beamforming and resource allocation, sensing-communication
performance tradeoffs, and standards-oriented discussion relevant to cellular or wireless ISAC. The
instructions excluded marketing material, vendor promotions, business or market analysis, and
speculative vision papers without technical explanation.
Generated Example Quality Judge.After local-vLLM example generation, this judge deter-
mines whether each generated training example should be retained. The default model was
meta-llama/Llama-3.3-70B-Instruct , with sampling temperature 0.2, top- p0.9 , maximum
output length 900 tokens, judge batch size 20, and tensor parallel size 2. Retained examples required
keep=true ,answerable_without_context=true , technical score at least 0.70, and clarity score
at least0.70.
Generated Example Quality Judge: system prompt
You are a strict dataset quality reviewer for ISAC tutorial data.
Generated Example Quality Judge: user prompt
Judge whether the following generated training example should be kept.
Reject if:
- It requires the original source text to understand (context-dependent, refers to ’this
paper/section/figure/table’).
- It is vague, hand-wavy, or mainly fluff.
- It contains factual errors or mismatched concepts (e.g., confusing radar Doppler with
carrier frequency offset).
- It is not about ISAC/JCAS or has no sensing+comm linkage.
- It includes citations, section numbers, or local references.
Return ONLY JSON with this exact shape:
{
"keep": true/false,
"answerable_without_context": true/false,
"technical_score": 0.0-1.0,
"clarity_score": 0.0-1.0,
"issues": ["..."]
}
EXAMPLE_TO_JUDGE_JSON:
17

<<<
{example_json}
>>>
Prompts used for filtering domain-specific Tele-Eval data.These prompts were used in the main
Tele-Eval source-document-split pipeline to filter QA pairs by target domain and align each retained
QA pair with supporting chunks from its source document.
Tele-Eval QA Domain Filtering Judge.This judge assigns each Tele-Eval QA pair domain relevance
scores for ISAC, SAGIN, and JCC. The default model was GPT-5.2, the temperature was 0.0, the
keep threshold was 0.65, the timeout was 60 seconds, and the maximum number of retries was 3. If
the OpenAI API key was unavailable in OpenAI mode, the pipeline fell back to keyword scoring.
Tele-Eval QA Domain Filtering Judge: system prompt
You are a strict telecom dataset curator. Return only three comma-separated numeric
scores.
Tele-Eval QA Domain Filtering Judge: user prompt
Judge whether this telecom QA pair belongs in a demo corpus focused on these domains:
{domain_lines}
Return ONLY three numbers between 0.0 and 1.0, in this exact order:
isac,sagin,jcc
Format requirements:
0.0,0.0,0.0
Do not return JSON.
Do not return prose.
Do not return labels, markdown, explanations, or units.
Each score is the QA pair’s relevance to that specific domain.
For each domain, 1.0 means clearly technical, answerable, and relevant to that domain.
For each domain, 0.0 means off-topic for that domain, generic, non-technical, malformed,
or lacking an answer.
QUESTION:
{question}
ANSWER:
{answer}
Contriever Alignment Chunk Judge.This judge scores candidate chunks from the matched source
document and selects the top positive passages used for InfoNCE training and Recall@ kevaluation.
The default model was GPT-5.2, the temperature was 0.0, the judge batch size was 8, the top positive
count was 3, the snippet length was 1800 characters, the timeout was 60 seconds, and the maximum
number of retries was 3. This chunk-alignment judge was not needed in the earlier generated-data
pipeline because each QA pair was generated from a known source chunk. In the Tele-Eval pipeline,
the source document is known, but the supporting chunks must still be aligned for supervised retriever
training and retrieval evaluation.
18

Contriever Alignment Chunk Judge: system prompt
You are a strict retrieval alignment judge. Return only comma-separated numeric scores.
Contriever Alignment Chunk Judge: user prompt
You are judging which document chunks were most likely used to generate a question-
answer pair.
Return ONLY {len(chunks)} comma-separated numbers between 0.0 and 1.0.
Do not return JSON, labels, markdown, prose, explanations, or units.
Score each chunk independently:
- 1.0 = the chunk directly supports the answer to the question
- 0.0 = the chunk is unrelated or insufficient
QUESTION:
{question}
ANSWER:
{answer}
CHUNKS:
{chunk_blocks}
Each chunk block was formatted as CHUNK {idx} | chunk_id={chunk_id} | vid={vid} fol-
lowed by the shortened chunk text.
Prompts used for filtering domain-specific TeleQnA questions.The domain-filtered TeleQnA
splits used an optional LLM judge after keyword/domain filtering. This judge scored whether a
multiple-choice QA item was substantively relevant to the target domain. The default model was
GPT-5.2, the temperature was 0.0, the maximum output length was 16 tokens, and the maximum
field length was 5000 characters. The confidence threshold was 0.70 in the original wrapper and 0.75
in the ARMOR 2.0 split-generation wrapper.
TeleQnA Domain Relevance Confidence Judge: system prompt
You are a strict wireless communications domain relevance classifier. You output only
one numeric confidence score.
TeleQnA Domain Relevance Confidence Judge: user prompt
Decide whether the following TeleQnA multiple-choice Q/A pair is relevant to the domain:
{domain_display_name}.
Target domain includes:
{domain_include}
Target domain excludes:
{domain_exclude}
Scoring:
- 1.0 = clearly and substantively relevant to the target domain.
- 0.0 = not relevant to the target domain.
- Use values between 0.0 and 1.0 for partial or uncertain relevance.
19

- Prefer precision over recall.
- Do not give high scores for generic telecom terms unless there is a clear domain
connection.
Return ONLY one floating-point number between 0 and 1.
No explanation. No JSON. No text.
QUESTION:
<<<
{question}
>>>
OPTIONS:
<<<
{options}
>>>
GOLD ANSWER:
<<<
{answer}
>>>
EXPLANATION:
<<<
{explanation}
>>>
Prompt used for evaluating open-ended Tele-Eval answers.This judge evaluates generated open-
ended answers by comparing each candidate response against the reference answer and returning a
scalar score. The default model was GPT-5.2, the temperature was 0.0, the maximum completion
length was 32 tokens, and failures returned score0.0.
Tele-Eval Answer Grading Judge: system prompt
You are an automatic grader for question-answer pairs. You output only one numeric score
.
Tele-Eval Answer Grading Judge: user prompt
You are grading a model answer for a question-answer pair.
Compare the candidate answer to the ground-truth reference.
Score semantics:
- 1.0 = on par with or better than the reference answer (technically correct, covers the
essential idea).
- <1.0 = worse than the reference.
Return ONLY one floating-point number between 0 and 1. No explanation, no text, no JSON.
Question:
<<<
{prompt}
>>>
Ground-truth answer:
<<<
{reference}
>>>
20

Candidate model answer:
<<<
{candidate}
>>>
B Capacity comparison for SFT and RAG fine-tuning
In this section, we derive generalization bounds for generator and retriever fine-tuning as a capacity
comparison. The purpose is to identify one estimation-complexity reason retriever adaptation can
be attractive in low-data domains, rather than to characterize all regimes in which retriever tuning
should outperform generator tuning.
B.1 General setup
We compare two families of adaptation,SFTandRAG, on the same population distribution Pover
examples z= (x, y) , where xis the question, and yis the target answer. Given Ni.i.d. training
pairs{zi}N
i=1drawn from the population distribution P, letθ∈ E S⊂RdSbe the trainable model
parameter for SFT, η∈ E R⊂RdRbe the trainable RAG parameters, and ℓS(z;θ) andℓR(z;η) be
the per-sample SFT and RAG loss, respectively.
B.1.1 Population and empirical losses
The population SFT loss is defined as the expected test loss under the true population distribution P,
while the empirical SFT loss is defined as the empirical training loss on finite sample{z i}N
i=1, i.e.
population loss:L S(θ) =E z∼P[ℓS(z;θ)],empirical loss: bLS,N(θ) =1
NNX
i=1ℓS(zi;θ).
The corresponding population and empirical risk SFT minimizer are defined as
population solution:θ∗∈arg min
θ∈ESLS(θ),empirical solution: bθ∈arg min
θ∈ESbLS,N(θ).
Similarly, we can define the population and empirical RAG losses and their corresponding optimal
solutions as follows.
population loss:L R(η) =E z∼P[ℓR(z;η)],empirical loss: bLR,N(η) =1
NNX
i=1ℓR(zi;η).
population solution:η∗∈arg min
η∈ERLR(η),empirical solution:bη∈arg min
η∈ERbLR,N(η).
A standard goal of generalization theory is to control the gap between the population loss and the
empirical loss. In our comparison, a smaller upper bound should be interpreted as better estimation
control under the stated assumptions rather than as a direct prediction of downstream performance.
B.1.2 Function class and Rademacher complexity
To state the comparison, we rewrite the objectives for SFT and RAG in terms of their backbone
prediction model class.
LetYdenote the answer label space and denote the RAG answer distribution as
qη(· |x) =X
d∈Nk(x)pη(d|x)p θ0(· |x, d)
whereN k(x)is the top-kdocument set ranked by the retrieval scores η(x, d) =f η(x)⊤g(d), and
pη(d|x) =exp (s η(x, d))P
d′∈Nk(x)exp (s η(x, d′)).
21

Therefore, we can write the backbone function class for SFT and RAG as
Fθ={x7→p θ(· |x) :θ∈ E S},F η={x7→q η(· |x) :η∈ E R}.(5)
Letting ℓNLL(q, x, y) =−logq(y|x) be the negative log-likelihood loss, the SFT and RAG loss can
be rewritten as
ℓS(z;θ) =ℓ NLL(pθ, x, y),andℓ R(z;η) =ℓ NLL(qη, x, y).
Rademacher complexity.Given the training samples ZN={z i}N
n=1, the empirical Rademacher
complexity of a function classFfor these samples is defined as follows
RZN(F) =1
NEϵ∼unif({1,−1})"
sup
f∈FNX
n=1ϵnf(xn)#
With Rademacher complexity, we have the following standard generalization bound.
Theorem 2.Assume that the loss function ℓNLL(q, x, y) is bounded in [0, c] and is ρ-Lipschitz
with respect to qin the feasible domain. Then, with probability at least 1−δ over the samples
ZN={z n}N
n=1, we have
sup
θ∈ESn
LS(θ)−bLS,N(θ)o
≤2ρR ZN(Fθ) + 3cr
log(2/δ)
2N(6)
sup
η∈ERn
LR(η)−bLR,N(η)o
≤2ρR ZN(Fη) + 3cr
log(2/δ)
2N(7)
Proof. The proof follows directly from the generalization bounds for machine learning models in
terms of Rademacher complexity, e.g. [28, Theorem 3.1], or [1, Theorem B.1].
According to Theorem 2, the second terms in the generalization bounds for SFT and RAG are the
same, so we will compare the Rademacher complexity RZ(Fθ)for SFT and RZ(Fη)for RAG in
the subsequent sections given the particular model architecture.
B.2 Generalization bound for generator fine-tuning
In this section, we specify the generator model as a Transformer-based model. For simplicity, we
analyze the one-layer Transformer model following [ 29], but extension to multi-layer Transformer is
also possible [ 36]. We denote the input for a single question-answer pair (x, y) after tokenization
asX∈RdS×d, and write the SFT parameter θ∈ E S⊂RdSasθ≜{W Q, WK, WV, WO}, where
WQ, WK, WV∈Rdm×d, WO∈Rdmare the query, key, value and output weight parameters in the
Transformer. Then the output of the Transformer can be expressed as
pθ(· |x) =W⊤
O 
1
dSdSX
i=1σr 
WVX⊤σs 
XW⊤
KWQ 
X(i,:)⊤
√dm!!!
∈R
where X(i,:)denotes the i-th row of tokenization matrix Xfor question-answer data pair (x, y) ,σs
denotes the row-wise softmax andσ rdenotes the ReLU activation function.
Following the lazy training regime in [ 29,36], we focus on the settings where the Transformer
parameters remain close to their initialization throughout training, i.e. the parameters are bounded.
Theorem 3([ 29, Lemma 1]).Suppose ∥Wt
V∥F≤R V,∥Wt
K∥F≤R K,∥Wt
Q∥F≤R Q,∥Wt
O∥F≤
ROholds for all iterations t. Also assume that the tokenized inputs for (xn, yn)have full rank and
∥Xn∥F≤√dSRXfor all n∈[N] , for some positive constant RX. The empirical Rademacher
complexity of the class of Transformer models Fθ={x7→p θ(· |x) :∥θ∥ ≤R} can be bounded
above by
RZN(Fθ)≤˜O r
PS
N3 
1 + log 
RORV(p
dSRX)r
N
PS!!!
≜C(N, R, R X, dS, dm)(8)
22

where ˜Ohides logarithmic dependencies exceptR, N,R=q
R2
V+R2
K+R2
Q+R2
O, and
PS= (p
dSRX)2p
dmRV2
3+p
dmRKRQRV2
33
log (Nd S).
Theorem 3 shows that the Rademacher complexity of the Transformer class grows with the parameter
dimensions dmanddS. In the low-data regime, where Nis small, this dimension-dependent term
can dominate the bound in Theorem 2, leading to a looser capacity upper bound for SFT.
B.3 Generalization bound for retriever fine-tuning
In this section, we also specify the retrieval model as a Transformer model, which is aligned with
the model we used in experiments. Similar to the SFT setting, we denote the input for a single
question xafter tokenization as XR∈RdR×d, and write the retrieval parameter η∈ E R⊂RdRas
η≜{U Q, UK, UV, UO}, where UQ, UK, UV∈Rdm′×d, UO∈Rdm′×dEare the query, key, value
and output weight parameters in the Transformer. The output of the Transformer can be expressed as
fη(x) =U⊤
O
1
dRdRX
i=1σr
UVX⊤
Rσs
XRU⊤
KUQ
X(i,:)
R⊤
√dm′


∈RdE
where X(i,:)
Rdenotes the i-th row of tokenization matrix XRfor question x,σsdenotes the row-wise
softmax andσ rdenotes the ReLU activation function.
Using the learned query encoderf η(x), the output of RAG is
qη(· |x) =X
d∈Nk(x)pη(d|x)p θ0(· |x, d)
whereN k(x)is the top-kdocument set ranked by the retrieval scores η(x, d) =f η(x)⊤g(d), and
pη(d|x) =exp (s η(x, d))P
d′∈Nk(x)exp (s η(x, d′)).
Theorem 4.Suppose ∥Ut
V∥F≤R V,∥Ut
K∥F≤R K,∥Ut
Q∥F≤R Q,∥Ut
O∥F≤R Oholds for all
iterations t. Also assume that the tokenized inputs for xnhave full rank and ∥XR,n∥F≤√dRRX
for all n∈[N] , for some positive constant RX. Additionally, assume ∥g(d)∥ ≤B gandqηis
ρsoft-Lipschitz over sη(x, d) on the feasible domain. The empirical Rademacher complexity of the
class of retrieval-augmented predictors Fη={x7→q η(· |x) :∥η∥ ≤R} can be bounded above by
RZN(Fη)≤˜O
ρsoftBgp
dEC(N, R, R X, dR, dm′)
(9)
where ˜Ohides logarithmic dependencies exceptR, N, andC(N, R, R X, dR, dm′)is defined in(8).
Proof. First, for each coordinate j∈[d E], let us denote the j-th coordinate of the query embedding
asfη,j(x)and define the corresponding query-embedding class as
Gη,j={x7→f η,j(x) :η∈ E R}
Then applying Theorem 3, we know that for anyj∈[d E],
RZN(Gη,j)≤˜O(C(N, R, R X, dR, dm′))(10)
Next, define the scalar retrieval score as
Sη,i={x7→f η(x)⊤g(di) :η∈ E R}
where i∈[k] anddi∈ D k(x). Then, since fη(x)⊤g(di) =PdE
j=1fη,j(x)g j(di), and by the
Cauchy-Schwarz inequality, we have
|fη(x)⊤g(di)| ≤B g
dEX
j=1fη,j(x)2
1/2
(11)
23

Therefore, according to the definition of Rademacher complexity, we have
RZN(Sη,i)≤B g
dEX
j=1RZN(Gη,j)2
1/2
≤˜O(B gp
dEC(N, R, R X, dR, dm′))(12)
Finally, applying Talagrand’s lemma for composite mappings [ 28, Lemma 4.2] to the softmax-
weighted generator mixture gives
RZN(Fη)≤˜O(ρsoftBgp
dEC(N, R, R X, dR, dm′))(13)
Remark 2.Because of top- kretrieval, the original qη(· |x) might not be continuous over η, so it
might not satisfy Lipschitz continuity over sη(x, d) . Following [ 3], we approximate qη(· |x) by a soft
surrogate with a fixed document poolD k(x)as
qη(· |x) =X
d∈Dk(x)pη(d|x)p θ0(· |x, d)
where
pη(d|x) =exp (s η(x, d))P
d′∈Dk(x)exp (s η(x, d′)).
With fixedkdocument pools for givenx,q ηcan be Lipschitz continuous overs η[3].
B.4 Comparisons between generator and retriever fine-tuning
Combining Theorems 2 and 4 yields Theorem 1. When the number of samples is small ( N≪d R, dS),
the empirical losses are comparable, and generator fine-tuning has many more trainable parameters
than retriever tuning ( dR≪d S), the empirical Rademacher complexity can dominate the comparison
between the two generalization upper bounds. In this regime, retriever tuning can have a smaller
bound because its lower-dimensional parameterization induces a smaller complexity term.
24