# Retrieval-Augmented Generation as Noisy In-Context Learning: A Unified Theory and Risk Bounds

**Authors**: Yang Guo, Yutian Tao, Yifei Ming, Robert D. Nowak, Yingyu Liang

**Published**: 2025-06-03 17:31:53

**PDF URL**: [http://arxiv.org/pdf/2506.03100v2](http://arxiv.org/pdf/2506.03100v2)

## Abstract
Retrieval-augmented generation (RAG) has seen many empirical successes in
recent years by aiding the LLM with external knowledge. However, its
theoretical aspect has remained mostly unexplored. In this paper, we propose
the first finite-sample generalization bound for RAG in in-context linear
regression and derive an exact bias-variance tradeoff. Our framework views the
retrieved texts as query-dependent noisy in-context examples and recovers the
classical in-context learning (ICL) and standard RAG as the limit cases. Our
analysis suggests that an intrinsic ceiling on generalization error exists on
RAG as opposed to the ICL. Furthermore, our framework is able to model
retrieval both from the training data and from external corpora by introducing
uniform and non-uniform RAG noise. In line with our theory, we show the sample
efficiency of ICL and RAG empirically with experiments on common QA benchmarks,
such as Natural Questions and TriviaQA.

## Full Text


<!-- PDF content starts -->

arXiv:2506.03100v2  [cs.LG]  4 Jun 2025Retrieval-Augmented Generation as Noisy In-Context Learning: A
Unified Theory and Risk Bounds
Yang Guo∗Yutian Yao†Yifei Ming‡Robert D. Nowak§Yingyu Liang¶
June 6, 2025
Abstract
Retrieval-augmented generation (RAG) has seen many empirical successes in recent years by aiding
the LLM with external knowledge. However, its theoretical aspect has remained mostly unexplored. In
this paper, we propose the first finite-sample generalization bound for RAG in in-context linear regression
and derive an exact bias-variance tradeoff. Our framework views the retrieved texts as query-dependent
noisy in-context examples and recovers the classical in-context learning (ICL) and standard RAG as
the limit cases. Our analysis suggests that an intrinsic ceiling on generalization error exists on RAG as
opposed to the ICL. Furthermore, our framework is able to model retrieval both from the training data
and from external corpora by introducing uniform and non-uniform RAG noise. In line with our theory,
we show the sample efficiency of ICL and RAG empirically with experiments on common QA benchmarks,
such as Natural Questions and TriviaQA.
1 Introduction
Retrieval-Augmented Generation (RAG) enhances language models by appending retrieved texts to the input,
enabling access to information beyond pretraining. It is widely used in open-domain QA, fact-checking, and
knowledge-intensive tasks [Huang et al., 2023, Lewis et al., 2020a, Ramos et al., 2022, Sarto et al., 2022, Zhao
et al., 2024a]. Retrieval sources typically fall into two categories: (1) labeled dataset , such as training dataset
itself [Liu et al., 2021, Izacard et al., 2022, Huang et al., 2024], and (2) generic corpora without labels , such
as Wikipedia [Chen et al., 2017]. Despite its promise, empirical studies show that increasing the number of
retrieved passages can degrade performance, especially when irrelevant or redundant texts are included [Levy
et al., 2025, 2024]. However, the theoretical aspects for understanding of how retrieval affects generalization
remain underexplored.
To study its behavior, we frame RAG as noisy in-context learning (ICL). ICL refers to the ability of
language models to adapt given the contextual information without updating model weights [Dong et al.,
2024]. Under this view, retrieved RAG examples can act as noisy context and its quality depends on the
retrieval. This view has motivated the development of many work in in-context retrieval [Luo et al., 2024,
Shi et al., 2022], where the goal is to retrieve high-quality demonstrate pairs, which reduces the noise of the
retrieval.
From a theoretical standpoint, RAG becomes tractable when framed as structured in-context learning,
where the context consists of fixed format demonstration pairs. Prior ICL work has analyzed this regime
under clean, i.i.d. examples [Ahn et al., 2023, Zhang et al., 2024]. These assumptions do not hold in RAG,
where retrieved examples are noisy, and their noise level tends to be inversely correlated to their relevance.
Currently, no theoretical framework has been developed to study RAG under this structured ICL formulation.
In this work, we bridge this gap by modeling RAG as noisy ICL, where retrieved examples follow a structured
∗yguo@cs.wisc.edu . University of Wisconsin-Madison.
†ytao37@wisc.edu . University of Wisconsin-Madison.
‡yifei.ming@salesforce.com . Salesforce AI Research.
§rdnowak@wisc.edu . University of Wisconsin-Madison.
¶yingyul@hku.hk . The University of Hong Kong. yliang@cs.wisc.edu . University of Wisconsin-Madison.
1

but perturbed distribution. In particular, we model the retrieval noise both under the uniform (same across
examples) and non-uniform (inversely correlated with the retrieval relevance). This view allows us to quantify
the impact of retrieval noise and derive generalization bounds that depend on the number of in-context and
RAG examples, and the retrieval distance from queries.
Our contributions are summarized as follows:
•We propose a theoretical framework for analyzing RAG and provide the first finite sample bounds for
in-context linear regression with RAG. Our bounds show that the improvement from RAG shrinks as
you add more retrieved examples, and can even flip to hurt performance, giving concrete guidance on
when to stop.
•Our framework includes ICL and standard RAG as limit cases, and also models retrieved data under
different noise regimes, uniform and non-uniform retrieval noise.
•We develop new tools for analyzing the query-dependent RAG data, e.g. a derivation of the expectation
for 6th-order Gaussian monomial (Lemma 3), which can be useful for future research on RAG.
•We conduct experiments for representative models on common QA datasets and demonstrate that
early RAG retrieves lie in the uniform noise regime, while later ones shift to non-uniform noise regime,
aligning with our theory.
2 Related Work
Retrieval Augmented Generation Retrieval-augmented generation (RAG) has emerged as a widely
adopted paradigm for enriching LLMs with external knowledge by prepending retrieved passages to the input
context [Lewis et al., 2020a, Izacard and Grave, 2020, Borgeaud et al., 2021]. From a functional perspective,
RAG transforms the model’s input distribution by conditioning generation on retrieved textual evidence,
often drawn from large-scale corpora via learned or heuristic retrieval mechanisms [Li et al., 2023, Meng
et al., 2024, Chen et al., 2024]. While much of the literature focuses on improving retrieval quality, system
performance [Asai et al., 2023, Li et al., 2024, Xu et al., 2024], and answer reliability [Xiang et al., 2024, Xu
et al., 2024], the theoretical foundations of RAG remain underexplored.
In-context Learning (ICL) ICL obtains its popularity from the original GPT-3 paper [Brown et al.,
2020], and becomes widely used in LLM applications [Dong et al., 2024, Min et al., 2021]. The recent
advance in ICL theory [Ahn et al., 2023, Zhang et al., 2024, Xie et al., 2021] provides a rigorous and versatile
framework to study transformers and LLMs. People have use this ICL framework to study novel setting,
like out-of-distributions tasks [Wang et al., 2024b] and test-time training [Gozeten et al., 2025]. People
also have also studied the noisy in-context learning from robustness [Cheng et al., 2025] and calibration
perspectives [Zhao et al., 2024b], which are different from our setup.
In-context Retrieval In-context retrieval [Luo et al., 2024] refers to retrieving a set of query-dependent
demonstrations than using fixed set of demonstrations. The label of the demonstration pairs can come from
various sources, such as in-domain training set [Izacard et al., 2022, Huang et al., 2024, Ye et al., 2023],
cross-domain data [Cheng et al., 2023, Shi et al., 2022], automatic LLM generation [Zhang et al., 2022, Li and
Qiu, 2023], pseudo-labels from unstructured data [Lyu et al., 2022, Li et al., 2022]. In our theoretical analysis
and experiments, we focus on the simplest in-context retrieval, in-domain retrieval from the training set, as
in [Izacard et al., 2022, Huang et al., 2024]. Note that in-context retrieval is a term developed later and some
earlier papers discuss ICL with retrieval as retrieving relevant documents without labels [Ram et al., 2023].
3 Problem Setup
Our problem setup is similar to [Zhang et al., 2024, Garg et al., 2022], with RAG examples to form the
additional in-context examples. It is worth noting that many works focus on ICL at test (inference) time,
specifically without parameter updates [Dong et al., 2022]. Our work adopts the framework of ICL with
2

warmup, also known as, supervised in-context training . Specifically, we assume that the pretraining data is
also formed by in-context examples. Then, during the test time, we formed prompts with in-context examples
with additional RAG examples.
Notations We denote [n] ={1, . . . , n }for an integer n≥1. We denote the trace product of two matrices
A, B∈Rm×nastr(AB⊤).
Pretraining Data We consider learning over linear regression data. The training data is a set of prompts.
Each prompt is of size m:(x1, y1, . . . ,xm, ym,xq)∈Rd(m+1)+ mwhere (x1, y1), . . . , (xm, ym)form the m
demonstration pairs. The goal is to predict ˆyqfor the query example xqto match the true label yq. The
prompt is embedded in the following form:
Ppt
m:=x1x2. . .xmxq
y1y2. . . y m 0
∈R(d+1)×(m+1)(1)
where (x1, y1), . . . , (xm, ym),(xq, yq)i.i.d.∼ D pt(pt denoting Pretraining). The output follows the linear model:
yi=x⊤
iβpt+ϵi, ϵ ii.i.d.∼ N (0, σ2)under Dpt (2)
where i∈[m]∪ {q},βttis the weight vector in pretraining, and ϵiis the noise for example i.
Inference Data (with RAG) During inference/test time, the test prompt Ptt+rag
m,n(tt denoting test-time) is
formed by min-context pairs (x1, y1), . . . , (xm, ym),nretrieval-augmented pairs (xrag
1, yrag
1), . . . , (xrag
n, yrag),
and the query pair xq, yq. The test prompt is embedded in the following form:
Ptt+rag
m,n :=x1. . .xmxrag
1. . .xrag
nxq
y1. . . y myrag
1 . . . yrag
n 0
∈R(d+1)×(m+n+1)(3)
The input xin each in-context or query pair follows the test-time distribution Dtt, and the label is:
yi=x⊤
iβtt+ϵi, ϵ ii.i.d.∼ N (0, σ2)under Dtt (4)
where i∈[m]∪ {q},ϵiis the noise of example i, and βttis the weight vector during test time. The input xin
each RAG pair follows the corresponding RAG distribution Drag(xq): assume the RAG query xrag
i=xq+ri
is generated around the query example xq, where riis the offset. The label in the RAG example is given by:
yrag
i= (xrag
i)⊤βtt+ϵrag
i, ϵrag
ii.i.d.∼ N (0, σ2
rag,i)under Drag(xq) (5)
where i∈[n],ϵrag
iis the noise of the i-th RAG example.
For the compactness of writing, we define the following matrices and vectors:
Xicl:= [x⊤
1;. . .;x⊤
m],Xrag:= [(xrag
1)⊤;. . .; (xrag
n)⊤],yicl:= [y1;. . .;ym],yrag:= [yrag
1;. . .;yrag
n],
ϵicl:= [ϵ1;. . .;ϵm],ϵrag:= [ϵrag
1;. . .;ϵrag
n],r= [r⊤
i;. . .;r⊤
n]
X=
Xicl
Xrag
∈R(m+n)×d,Xrag=
xq+r1
...
xq+rn
∈Rn×d,y=yicl
yrag
∈ Rm+n,ϵ=
ϵicl
ϵrag
∈ Rm+n
Training and Testing We let Wbe the model parameters, and Fbe the model. Given an input prompt
Ppt
mwith demonstration pairs, the model predicts ˆyq:=F(Ppt
m;W). As a common practice in theoretical
studies of LLM for feasible analysis, we use the MSE loss as the evaluation metrics [Zhang et al., 2024, Ahn
et al., 2023, Xie et al., 2021]. Then, the population loss on the pretraining data is:
Lpt(W) := E
(x1,y1),...,(xm,ym)(xq,yq)∼Dpth 
yq−F 
Ppt
m;W2i
(6)
3

Its minimizer is denoted as:
¯W∗:= min
WLpt(W). (7)
To apply the pretrained W∗from the pretraining context size of mto the test-time context size of m+n,
we will need to scale it properly (see Lemma 1) and use
W∗=m
m+n¯W∗. (8)
During the test time we evaluate the population loss over the test prompt with RAG examples Ptt+rag
m,n:
Ltt+rag (W) := E
(x1,y1),...,(xm,ym)(xq,yq)∼Dtt
(xrag
1,yrag
1),...,(xrag
n,yrag
n)∼Drag(xq)h 
yq−F 
Ptt+rag
m,n ;W2i
(9)
Model Architecture We study the single-layer linear self-attention model (LSA) as the framework for
theoretical analysis, similar to many existing studies (e.g., [Ahn et al., 2023, Zhang et al., 2024]). The
prediction of the model Fon a prompt Pwith query xqis:
ˆyq:=F(P) = [PWQW⊤
KP⊤PWV]m+n+1,d+1=x⊤
qWX⊤y (10)
where the query, key, and value matrices WQ,W,WV∈R(d+1)×(d+1)are parameterized by Win the follow
way:
WQW⊤
K=W 0d×1
01×d 0
,WV=0d×d0d×1
01×d 1
We note that this parameterization is commonly used in the previous works [Ahn et al., 2023, Zhang et al.,
2024], and is shown to capture the key properties of in-context learning. Furthermore, [Ahn et al., 2023]
shows that the formulation is the optimum converged from pretraining on Gaussian data.
4 Theoretical Analysis: Generalization Bound for RAG
To study test-time error and sample complexity in in-context linear regression with RAG examples, we
consider two noise regimes: uniform retrieval noise andnon-uniform retrieval noise . Uniform retrieval
noise assumes the RAG noise ϵrag
ifor each example iis i.i.d. Since its variance is distance-agnostic, it can
model a scenario of retrieval where the noise is prevailing across data points. Non-uniform retrieval noise
assumes either the variance or the label-corruption probability grows with the variance of retrieval vector —
e.g.σ2
rag,iincreases with δ2
ior probability of making mistakes increases with δ2
i. This captures retrieval from
datasets where near neighbors often supply the right signal while far ones are potentially noisy or misleading.
Because the noise spectrum is now heavy-tailed, adding more RAG examples past a threshold could yield
diminishing benefits for RAG examples and even become counter-productive. Framing RAG through these
two lenses allows precise clarification about when extra retrieved examples will pay off, and when they will
hit the intrinsic ceiling and more retrieved examples don’t help anymore. These are well corroborated by our
experimental results on real data (see Section 5).
First, we introduce the key data assumptions.
Assumption 1 (Gaussian Retrieval Offset) .We assume the retrieval offset ri,∀i∈[n]to follow a Gaussian
distribution: rii.i.d.∼ N 
0, δ2
iId
.
The key property that we want to control for RAG examples is its distance from the query points xq.
However, modeling the queried example directly through the retrieval distance leads to complicated theoretical
analysis. Here, we note that the retrieval distance ∥ri∥2converges to a distribution concentrated in an
O(δi√
d)ball around the query with respect to d[Cover and Hart, 1967]. Thus, controlling the variance of
the retrieval offset can alternatively control the retrieval distance. And we make the following additional data
assumptions.
Assumption 2 (Data Assumption) .We assume the data follows the following:
4

1.Pretraining Examples ( Dpt). For a pretraining prompt of length m+ 1and for all i∈[m]∪ {q},
we assume xii.i.d.∼ N (0, I),ϵii.i.d.∼ N (0, σ2),βpt∼N(0, I).
2.Test Time Examples ( Dtt). For a test-time prompt of length m+n+ 1and for all i∈[m]∪ {q}, we
assume xii.i.d.∼ N (0, I),ϵii.i.d.∼ N (0, σ2),βtt∼N(0, I).
3.Test-Time RAG Examples ( Drag(xq)). For a test-time prompt of length m+n+ 1and for all
i∈[m+ 1, . . . , m +n], we assume xrag
ii.i.d.∼ N (0, I),ϵrag
i∼N(0, σ2
rag,i), and the same βttas (2).
Here, we assume the isotropic Gaussian property for the input, noise and the weight vector, a common
assumption made in ICL theory [Ahn et al., 2023, Gozeten et al., 2025] for simple yet meaningful analysis.
Overview of the Key Results
•(Uniform Noise ) RAG examples are as effective as ICL examples in reducing the variance-induced err
but ineffective at reducing the bias-induced err, causing a loss plateau for n→ ∞,
•(Non-Uniform Noise ) RAG could improve the variance-induced error up to a finite nat a cost of
increasing bias-induced error.
Roadmap Under these assumptions and uniform retrieval noise, we will first derive the population loss of
RAG, Ltt+rag (W), for general Was in Theorem 1, analyze its finite sample complexity under the optimal
pretrained weight W∗as in Proposition 1 and derive an optimal number of RAG examples of n∗for a given
number of ICL examples mas in Proposition 2. These discussions leads to our first key result. Then, under
the non-uniform retrieval noise, we will prove the sample complexity under the distance-proportional noise
(Theorem 2) and distance-weighted mixture noise (Theorem 3), and obtain our second key results above.
4.1 Uniform Retrieval Noise
Assumption 3 (Uniform Retrieval Noise) .The RAG noise ϵragshares the same Gaussian distribution with
variance σ2
rag, i.e.∀i∈[m+ 1, . . . , m +n],σ2
rag, i =σ2
rag.
First, we present the assumption for uniform retrieval noise. In other words, all RAG examples are as
helpful, and its improvement on the actual prediction is determined by the retrieval distance.
Theorem 1 (Population Loss for ICL with RAG Examples) .Under Assumption 1, 2, 3, the population loss
of the linear self-attention predictor ˆyq=x⊤
qWX⊤ysatisfies
Ltt+rag (W) =E(E(ˆyq)−ˆyq)2
|{z }
:=err variance (W)+E(E(ˆyq)−E(yq))2
| {z }
:=err bias(W)+ σ2
|{z}
irreducible noise, and specifically, (11)
errvariance (W) =
mσ2+ 
1 +δ2
nσ2
rag
tr(W⊤W) +nσ2
ragtr(W2) +nσ2
ragtr(W)2
errbias(W) =β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I+M4i
βtt
=β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I
+
n2 
2 +δ2
+n 
m+δ2
W2+ 
W2⊤
+ 2n(n+δ2)WW⊤
+
m2+m+mn 
2 + 2 δ2
+n2 
2 + 2 δ2+δ4
+n 
2δ2+δ4
W⊤W
+
n2 
2 +δ2
+n 
m+δ2
tr(W)
W+W⊤
+
n2+nδ2 
tr(W)2+ tr 
W2
I+
m+n2+n 
2δ2+δ4
tr
W⊤W
Ii
βtt
Here, we derive the exact bias-variance decomposition for ICL with RAG. The first line is the variance-
induced error formed by a weighted sum of noise from ICL examples and RAG examples. Because of the
implicit scaling of Was discussed in Lemma 1, the second order term in Wwill introduce an additional
5

weight scaling ofm2
(m+n)2when adapting from the weight learned on msize context to m+nsize context.
Thus, larger nwill let errvariance (W)→0, and the convergence rate is affected by δ2. Larger retrieval distance
leads to a slower convergence. The bias-induced error is composed of all possible monomials of Wup to the
2nd-order with troperation. The complex dependency on m, n, δ2, drequires additional assumptions on W
to further interpret. As a sanity check, when n= 0(ICL-only), this decomposition can exactly recover loss as
in Lemma B.2 in [Gozeten et al., 2025].
As a proof sketch, we first compute errvariance (W) =E(x⊤
qWX⊤ϵ)2by splitting the calculation for ICL
and RAG examples based on X. Then, we compute errbias(W) =E[(x⊤
q(I−WX⊤X)βtt)2]. The main
technical challenge lies in the dependency of Xragonxq, and errbiashas a 6th-order dependency on xq(2
fromxqand 4 from X). As shown in Lemma 3, E
xqx⊤
qAxqx⊤
qBxqx⊤
q
gives 15 new terms that include all
the second order monomials of Wwith tr. The calculation requires multiple careful applications of Isserlis’
theorem [Isserlis, 1918], and the full proof can be seen in Appendix B. It is possible to prove this theorem
for a design matrix with non-isotropic covariance, but computing the expectation of the 6th-order Gaussian
monomial is more complicated.
Here, we present the finite sample bound for pretrained W∗for better interpretation.
Proposition 1 (Finite Sample Generalization Bound) .Under Assumption 1, 2, 3, if δ2≪1,
Ltt+rag (W∗) =O
σ2+dm
(m+n)2σ2+d2n
(m+n)2σ2
rag
| {z }
errvariance (W∗)+∥βtt∥2
2"
d
m+d2n
m+n2#
| {z }
errbias(W∗)

errvariance (W∗) =

O(d
mσ2+d2
m2σ2
rag) =O 1
m
m→ ∞,nfixed.
O(d
n2σ2+d2
nσ2
rag) =O 1
n
n→ ∞,mfixed
O(d
mσ2+d2
mσ2
rag) =O 1
m
m, n→ ∞,n= Θ( m)(12)
errbias(W∗) =

O 
∥βtt∥2
2d
m
ifm→ ∞,nis fixed
O 
∥βtt∥2
2d2
=C1 ifn→ ∞,mis fixed
O 
∥βtt∥2
2 d
m+d2
=C2+O(∥βtt∥2
2d
m)ifm→ ∞,n= Θ( m)(13)
Here, we assume δ2≪1as the test time example xihas only a variance of I, and it is unrealistic to
assume a higher retrieval variance than the input variance. On the limit case where m→ ∞andnare fixed,
we observe that both variance-induced and bias-induced error decay at a rate of O(1/m), matching the
results from the existing paper [Ahn et al., 2023, Zhang et al., 2024]. When n→ ∞, the variance-induced
error decays as O(1/n)matching the O(1/m)rate. However, introducing the RAG is ineffective at reducing
the bias-induced error. Even when m→ ∞, increasing nwill cause a loss plateau.
This effect can be explained by the underlying adaptive ability of transformers. In an online learning
setup, we could always use the mean of the queried data as the prediction. However, in the LSA setup, the
pretrained W∗serves as a proxy for E−1(X⊤X). In order to retain the adaptivity to the entire distribution
ofβtt, we cannot use the optimal linear classifier (X⊤X)−1X⊤yor use the mean of the retrieved examples
ad hoc. At the test stage, Xragonly appears in X⊤yand not in W∗. The difference between Drag(xq)and
Dttdirectly leads to the increase of variance worsened by the increase of n. See full proof in Appendix B.
Now, a natural question is whether we can find a balance of variance and bias and obtain an optimal RAG
example size n∗.
Proposition 2. Under Assumption 1,2,3, δ2≪1, and reasonable choice of σ2, σ2
rag(σ2, σ2
rag≪ ∥βtt∥2
2), the
optimal n∗that minimizes the RAG loss follows:
n∗=O 
m 
d2∥βtt∥2
2+dσ2−d2σ2
rag
md2∥βtt∥2
2−d2σ2rag!
=O 
d∥βtt∥2
2+σ2−dσ2
rag
d∥βtt∥2
2!
(14)
and the improvement on loss from picking the optimal n∗over n= 0is given as:
Ltt+rag (W∗)|n=0− Ltt+rag (W∗)|n=n∗=O1
m2
(15)
6

In fact, the optimal n∗does not scale with momitting the lower-order terms. Note that for ∥βtt∥2
2=O(1),
∥βtt∥2
2will dominate the numerator for reasonable choices of σ2andσ2
rag. A larger ICL noise σ2leads to a
larger n∗, i.e. requiring more RAG examples to compensate for the loss. A larger RAG noise σ2
ragleads to a
smaller n∗, i.e. less efficiency on RAG examples. And the improvement converges at O(1
m2), diminishing for
large m. See the full proof in Appendix B. Several empirical works also observe a performance drop when
increasing the number of retrieved examples [Wang et al., 2024a, Levy et al., 2025].
4.2 Non-Uniform Retrieval Noise
The uniform-noise setup in Section 4.1 relies on a clean retrieval pool, so we could keep the variance σ2
rag
fixed. In open-domain retrieval, this assumption could collapse: many retrieved examples could contain no
answer or even a wrong answer. Empirically, people have observed that passages that are closer to the query
vector xqare more likely [Yang and Seo, 2020, Yoran et al., 2023, Lewis et al., 2020b] to contain the correct
label. We want to theoretically investigate if the following hypothesis still holds:
Closer to query xq=⇒more likely to contain correctanswer.
4.2.1 Distance-Proportional Noise (DPN)
We first investigate the scenario where the retrieval noise is proportional to the retrieval distance. Since the
ICL analysis only applies to the mean-squared error loss, we study the effect of RAG under DPN on the
correctness of the predictions.
Assumption 4 (Distance-Proportional Noise) .There exists a constant γ1>0such that, for every retrieved
sample i,σ2
rag,i=γ1σ2δ2
i,i.e. the RAG noise variance grows linearly with the variance δ2
ithat governs the
retrieval distance.
Under the new data assumption, we denote the corresponding RAG loss, bias-induced error, and variance-
induced error for Wto be ˆLtt+rag (W),ˆ errbias(W), and ˆ errvariance (W).
Theorem 2 (Finite Sample RAG Generalization Bound under DPN) .Under Assumption 1, 2, 4, the
population loss is given as:
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1γ1δ2
i[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
If the variance of the retrieval distance follows power law, i.e. ∃γ2>0, q≥0s.t.δ2
i=γ2iq, then
ˆ errbias(W∗) =O
errbias(W∗) +∥βtt∥2
2dn2q+1+n2q+2
(m+n)2
(16)
and
ˆ errvariance (W∗) =Odmσ2+d(n2q+1)σ2
(m+n)2
=(
O 
dn2q−1σ2
ifn→ ∞,q≤1/2
diverges if n→ ∞,q >1/2(17)
Here, we derive the sample complexity under DPN. A second order dependency on δ2
ishows up in both
the variance-induced and bias-induced error (exact form seen in Appendix B). Thus, the δ2
i-involved constant
will dominate the other constants. Specifically, it even leads to divergence for q >1/2for the variance-induced
error and q >0for the bias-induced error.
4.2.2 Distance-Weighted Mixture Noise
In this section, we discuss the scenario where further RAG examples are less likely to contain the correct
answers. We use a pair of large and small noises to model the correct/incorrect examples.
7

Assumption 5 (Distance-Weighted Mixture Noise) .We assume that the RAG noise is formed by a mixture
of small and large noise:
y(xrag) =(
f(xrag,i) +ϵsw.p. pi
f(xrag,i) +ϵlw.p. 1−pi
where ϵs∼ N(0, csσ2)corresponds to the small noise and ϵl∼ N(0, clσ2)corresponds to the large noise, with
cl≥cs≥0. The probability of sampling small noise pifollows an inverse power law of the variance of the
retrieval distance, i.e. pi= (1 + δ2
i)−˜q,˜q≥0.
Here, we choose the sampling probability (of small noise) pito follow a polynomial decay and the constant
1 here is to ensure pi= 0when δ2
i= 0. Under the new data assumption, we denote the corresponding RAG
loss, bias-induced error, and variance-induced error for Wto be ˜Ltt+rag (W),˜ errbias(W), and ˜ errvariance (W).
Theorem 3 (Finite Sample RAG Bound under Distance-Weighted Mixture Noise) .Under Assumption 1, 2,
5, then ˜ errbias(W) = ˆ err bias(W), and
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1 
piσ2
s+ (1−pi)σ2
l
[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
If the variance of the retrieval distance follows power law, i.e. ∃γ2>0, q≥0s.t.δ2
i=γ2iq, then:
˜ errvariance (W∗) =(
O 
cldnq−1σ2−(cl−cs)σ2dnq−1−q˜q
ifn→ ∞,q≤1
diverges if n→ ∞,q >1(18)
The bias-induced error here is the same as in DPN, since we assume a polynomial dependency for δ2
ioni
in both setting and the bias-induced error is independent of the variance of noise. Even though the variance
of small/large noise are bounded, the dependency on the retrieval distance leads to the divergence at large q
(q >1). The large prediction noise will dominate the variance-induced error, but a larger gap between large
and small noise ( cl−cs) can mitigate the error by a ratio of O(n−q˜q). That is, the smaller qand˜qare, the
lower the error.
We note that the uniform noise scenario can also admit the mixture noise model by taking a constant
pi,∀i, resulting in a form similar to the standard uniform retrieval noise in Proposition 1.
5 Experiments
We investigate the effect of RAG focusing on the following questions: ( Q1) Whether RAG data outperform
randomly sampled in-context examples? ( Q2) What are the impacts of the RAG examples from training
data and RAG passages from external corpora? ( Q3) With a fixed budget of example numbers, what is
the effect of varying the ratio between the two types of RAG data? Our experiments provide the following
findings: ( A1) RAG data lead to better performance than in-context ones under different data budgets.
(A2) Interestingly, the first few RAG training examples significantly improve performance, but later ones are
harmful, because the first few are highly relevant but later ones are noise rather than signal. In contrast,
RAG passages from external corpora can slowly but monotonically improve the performance, because external
corpora are large enough to provide noisy but still relevant data. These are captured by different noise models
in our theory. (A3)The performance is not monotonic with the ratio, and the sweet spot depends on the
data/model.
Setup For Natural Questions (NQ), the retrieval index is constructed from the December 2018 Wikipedia
dump. For TriviaQA, we use the December 2021 version. To accommodate hardware limitations, we randomly
subsample 10% of the full index for both datasets. This reduces retrieval cost and memory usage, allowing all
experiments to be conducted on a single NVIDIA A100 or L40 GPU.
We use representative models ATLAS Izacard et al. [2022] and RAVEN Huang et al. [2024] on two
standard open-domain question answering benchmarks Natural Questions (NQ) Kwiatkowski et al. [2019]
andTriviaQA Joshi et al. [2017]. For evaluation, the context consists of min-context examples, and n
RAG data points (including n1RAG examples from the training data and n2RAG passages from external
8

corpora like Wikipedia, so n=n1+n2). We choose different m, n 1, n2’s for our study purpose and report
the standard exact match (EM) accuracy on 1000 random samples from the test set.
RAG v.s. In-Context For a budget c, we compare using RAG only ( m= 0, n1=n2=c/2) and in-context
examples only ( m=c, n= 0). The results in Figure 1 show that RAG consistently outperforms in-context
examples, as RAG provides query-relevant data with more signals to address the query, consistent with our
analysis.
4 10 16 22 28 34 40
T otal Number of Data Points c020406080100Exact Matc  Accuracy (%)NQ da(ase(
Inc%n(ex(-%nly (A TLAS)
R AG-%nly (A TLAS)
Inc%n(ex(-%nly (R A VEN)
R AG-%nly (R A VEN)
4 10 16 22 28 34 40
T otal Number of Data Points c020406080100Exact Matc  Accuracy (%)T riviaQA da(ase(
Inc%n(ex(-%nly (A TLAS)
R AG-%nly (A TLAS)
Inc%n(ex(-%nly (R A VEN)
R AG-%nly (R A VEN)
Figure 1: We compare performance between the RAG-only ( c=m) versus in-context-only methods ( c=
n1+n2, n1=n2), where cis the total number of data, n1refers to retrieved examples and n2to passages.
0 10 20 30 40
T otal Number of Data P o nts c020406080100Exact Match Accuracy (%)NQ da(ase(
R AG %assages (A TLAS)
R AG exam%les (A TLAS)
R AG %assages (R A VEN)
R AG exam%les (R A VEN)
0 10 20 30 40
T otal Number of Data P o nts c020406080100Exact Match Accuracy (%)T riviaQA  dataset
R AG passages (A TLAS)
R AG examples (A TLAS)
R AG passages (R A VEN)
R AG examples (R A VEN)
Figure 2: We compare the performance of RAG using examples ( c=n1) versus passages ( c=n2).
RAG Examples v.s. RAG External Passages Next, we compare using RAG examples from training
data only ( m= 0, n1=c, n2= 0) and RAG passages from external corpora only ( m= 0, n1= 0, n2=c). The
results in Figure 2 show interesting patterns. For RAG examples only, with more examples, the performance
first significantly improves but later drops. This suggests that the first few examples are highly relevant but
later ones contain more noise than signal. In contrast, for RAG passages only, the performance increases more
slowly but steadily for larger budgets. This suggests the passages retrieved are noisy but still have relevant
9

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
F raction o  R etrieved Examples n1/c010203040506070Exact Match Accurac. (%)A TLAS o% NQ data)et
total %umbe( o  data poi%t) c=10
total %umbe( o  data poi%t) c=16
total %umbe( o  data poi%t) c=22(a) ATLAS Performance as a function of n1/cunder
different data points con NQ.
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
F raction of Retrieved Examples n1/c010203040506070Exact Matc  Accuracy (%)A TLAS %n T riviaQA da)a(e)
)%)al number %f da)a p%in)( c=10
)%)al number %f da)a p%in)( c=16
)%)al number %f da)a p%in)( c=22(b) ATLAS Performance as a function of n1/cunder
different data points con TriviaQA.
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
F raction  of R etr eved Examples n1/c010203040506070Exact Match Accuracy (%)R A VEN on NQ da(ase(
(o(al n)mber of da(a %o n(s c=10
(o(al n)mber of da(a %o n(s c=16
(o(al n)mber of da(a %o n(s c=22
(c) RAVEN Performance as a function of n1/cunder
different data points con NQ.
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
F raction of Retrieved Examples n1/c010203040506070Exact Matc  Accuracy (%)R A VEN %n T riviaQA da)a(e)
)%)al number %f da)a p%in)( c=10
)%)al number %f da)a p%in)( c=16
)%)al number %f da)a p%in)( c=22(d) RAVEN Performance as a function of n1/cunder
different data points con TriviaQA.
Figure 3: Performance sensitivity to the ratio n1/nunder different data points c, where n1refers to retrieved
examples and n2to passages.
signals. This aligns with our noise modeling. When n1is small ( ≤20for NQ and ≤10for TriviaQA), RAG
examples resemble uniform noise due to the relevance of retrieved examples. As n1increases, n1introduces
more irrelevant or conflicting examples (i.e., non-uniform noise ). On the other hand, n2resembles a uniform
noiseregime as the retrieval pool is broad with relevant data but also noisy.
When the retrieval budget is small, retrieval from training examples yield higher accuracy than from
passages, even though both operate in the uniform-noise regime. This discrepancy follows from the mixture-
noise effects: a passage judged relevant may still lack any answer-bearing text, raising its effective noise level
relative to examples. Furthermore, the significant drop for the retrieval from examples as opposed to retrieval
from passages can be explained by the size difference for the training data and passages pool (i.e. Wikipedia).
Since the passages provide a denser coverage of the semantic space, more passages will remain relevant as
opposed to examples. In all, our theory covers both practical data types and matches the empirical results.
Ratio between RAG Examples and Passages The different noise properties of the two kinds of RAG
data imply that we should find a proper ratio between them when the total budget cis fixed. Figure 3 in the
appendix shows that as the ratio n1/cincreases, the performance initially improves—benefiting from signal
information—but eventually declines as low-quality examples dominate the context. This again supports
10

our theoretical view of signal versus noise in the retrieved data. The results demonstrate that performance
initially improves as more signal (examples) is added, but eventually declines due to increasing noise from
irrelevant or low-quality examples. This supports the theoretical perspective of balancing signal and noise in
retrieval-augmented inputs.
6 Conclusion and Limitations
We model RAG as query-dependent noisy in-context learning, derive the first finite-sample error bounds
for linear regression that isolate the contributions of retrieval signal and noise, and extend those bounds to
different noise regimes and test-time training. Experiments on Natural Questions, TriviaQA with RAVEN,
and ATLAS corroborated our theoretical analysis.
Regarding limitations, our bounds focus on the linear setting, opening avenues for future studies on
nonlinear methods like kernels and neural networks. While our framework accounts for common RAG noise
models, new models may be needed for other types of RAG data. A further direction is to combine RAG
with test-time training, studying how on-the-fly adaptation affects both theoretical guarantees and empirical
performance. Our experiments feature representative models and datasets, but future research can explore
newer retrievers, LLMs like Qwen 3 and Llama 4, and more advanced RAG applications.
7 Acknowledgment
This work is partially supported by the National Science Foundation (NSF) under Grant CCF-2046710. I
also thank my colleague Haoyue Bai for insightful discussions.
11

References
Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement
preconditioned gradient descent for in-context learning. Advances in Neural Information Processing
Systems, 36:45614–45650, 2023.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve,
generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511 , 2023.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, T. W. Hennigan, Saffron Huang, Lorenzo Maggiore, Chris Jones, Albin
Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan,
Jack W. Rae, Erich Elsen, and L. Sifre. Improving language models by retrieving from trillions of tokens.
InInternational Conference on Machine Learning , 2021.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in neural information processing systems , 33:1877–1901, 2020.
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer open-domain
questions. arXiv preprint arXiv:1704.00051 , 2017.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-
lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In
Annual Meeting of the Association for Computational Linguistics , 2024.
Chen Cheng, Xinzhi Yu, Haodong Wen, Jingsong Sun, Guanzhang Yue, Yihao Zhang, and Zeming Wei.
Exploring the robustness of in-context learning with noisy labels. In ICASSP 2025-2025 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1–5. IEEE, 2025.
Daixuan Cheng, Shaohan Huang, Junyu Bi, Yuefeng Zhan, Jianfeng Liu, Yujing Wang, Hao Sun, Furu Wei,
Denvy Deng, and Qi Zhang. Uprise: Universal prompt retrieval for improving zero-shot evaluation. arXiv
preprint arXiv:2303.08518 , 2023.
Thomas Cover and Peter Hart. Nearest neighbor pattern classification. IEEE transactions on information
theory, 13(1):21–27, 1967.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang
Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234 , 2022.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu,
Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. A survey on in-context learning. In Proceedings of the
2024 Conference on Empirical Methods in Natural Language Processing , pages 1107–1128, Miami, Florida,
USA, November 2024. Association for Computational Linguistics.
Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context?
a case study of simple function classes. Advances in Neural Information Processing Systems , 35:30583–30598,
2022.
Halil Alperen Gozeten, M Emrullah Ildiz, Xuechen Zhang, Mahdi Soltanolkotabi, Marco Mondelli, and
Samet Oymak. Test-time training provably improves transformers as in-context learners. arXiv preprint
arXiv:2503.11842 , 2025.
Jie Huang, Wei Ping, Peng Xu, Mohammad Shoeybi, Kevin Chen-Chuan Chang, and Bryan Catanzaro.
Raven: In-context learning with retrieval-augmented encoder-decoder language models, 2024. URL
https://arxiv.org/abs/2308.07922 .
12

Rongjie Huang, Jia-Bin Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu,
Xiaoyue Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion
models.ArXiv, abs/2301.12661, 2023.
Leon Isserlis. On a formula for the product-moment coefficient of any order of a normal frequency distribution
in any number of variables. Biometrika , 12(1/2):134–139, 1918.
Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain
question answering. ArXiv, abs/2007.01282, 2020.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented
language models, 2022. URL https://arxiv.org/abs/2208.03299 .
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of
the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages
1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/
P17-1147. URL https://aclanthology.org/P17-1147/ .
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A
benchmark for question answering research. Transactions of the Association for Computational Linguistics ,
7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026/ .
Mosh Levy, Alon Jacoby, and Yoav Goldberg. Same task, more tokens: the impact of input length on the
reasoning performance of large language models. arXiv preprint arXiv:2402.14848 , 2024.
Shahar Levy, Nir Mazor, Lihi Shalmon, Michael Hassid, and Gabriel Stanovsky. More documents, same
length: Isolating the challenge of multiple documents in rag. arXiv preprint arXiv:2503.04388 , 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-
augmented generation for knowledge-intensive nlp tasks. In Proceedings of the 34th International Conference
on Neural Information Processing Systems , NIPS ’20, Red Hook, NY, USA, 2020a. Curran Associates Inc.
ISBN 9781713829546.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in neural information processing systems , 33:9459–9474, 2020b.
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. Making large language models a better foundation
for dense retrieval. ArXiv, abs/2312.15503, 2023.
Junlong Li, Jinyuan Wang, Zhuosheng Zhang, and Hai Zhao. Self-prompting large language models for
zero-shot open-domain qa. arXiv preprint arXiv:2212.08635 , 2022.
Xiaonan Li and Xipeng Qiu. Mot: Pre-thinking and recalling enable chatgpt to self-improve with memory-of-
thoughts. CoRR, 2023.
Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing.
Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous
sources. In The Twelfth International Conference on Learning Representations , 2024.
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good
in-context examples for gpt- 3?arXiv preprint arXiv:2101.06804 , 2021.
Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, and Mehran Kazemi. In-context learning with retrieved
demonstrations for language models: A survey. arXiv preprint arXiv:2401.11624 , 2024.
13

Xinxi Lyu, Sewon Min, Iz Beltagy, Luke Zettlemoyer, and Hannaneh Hajishirzi. Z-icl: Zero-shot in-context
learning with pseudo-demonstrations. arXiv preprint arXiv:2212.09865 , 2022.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfr-embedding-
mistral:enhance text retrieval with transfer learning. Salesforce AI Research Blog, 2024. URL https:
//blog.salesforceairesearch.com/sfr-embedded-mistral/ .
Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context.
arXiv preprint arXiv:2110.15943 , 2021.
Kaare Brandt Petersen, Michael Syskind Pedersen, et al. The matrix cookbook. Technical University of
Denmark , 7(15):510, 2008.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computa-
tional Linguistics , 11:1316–1331, 2023.
Rita Parada Ramos, Bruno Martins, Desmond Elliott, and Yova Kementchedjhieva. Smallcap: Lightweight
image captioning prompted with retrieval augmentation. 2023 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , pages 2840–2849, 2022.
Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Retrieval-augmented transformer for
image captioning. Proceedings of the 19th International Conference on Content-based Multimedia Indexing ,
2022.
Peng Shi, Rui Zhang, He Bai, and Jimmy Lin. Xricl: Cross-lingual retrieval-augmented in-context learning
for cross-lingual text-to-sql semantic parsing. arXiv preprint arXiv:2210.13693 , 2022.
Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu,
Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context llms with extended
multi-doc qa. arXiv preprint arXiv:2406.17419 , 2024a.
Qixun Wang, Yifei Wang, Yisen Wang, and Xianghua Ying. Can in-context learning really generalize to
out-of-distribution tasks? arXiv preprint arXiv:2410.09695 , 2024b.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek Mittal. Certifiably robust
rag against retrieval corruption. arXiv preprint arXiv:2405.15556 , 2024.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning
as implicit bayesian inference. arXiv preprint arXiv:2111.02080 , 2021.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. RECOMP: Improving retrieval-augmented LMs with context com-
pression and selective augmentation. In The Twelfth International Conference on Learning Representations ,
2024.
Sohee Yang and Minjoon Seo. Is retriever merely an approximator of reader? arXiv preprint arXiv:2010.10999 ,
2020.
Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. Compositional exemplars for in-context
learning. In International Conference on Machine Learning , pages 39818–39833. PMLR, 2023.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models
robust to irrelevant context. arXiv preprint arXiv:2310.01558 , 2023.
Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. Journal
of Machine Learning Research , 25(49):1–55, 2024.
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large
language models. arXiv preprint arXiv:2210.03493 , 2022.
14

Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao
Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A survey. arXiv
preprint arXiv:2402.19473 , 2024a.
Yufeng Zhao, Yoshihiro Sakai, and Naoya Inoue. Noisyicl: A little noise in model parameters calibrates
in-context learning. arXiv preprint arXiv:2402.05515 , 2024b.
15

A Technical Preliminaries
Additional Notations For two integer indices iandj, we denote δij=(
1ifi=j
0ifi̸=jas Kronecker delta.
Lemma 1 (Adapt Wto Different Context Size) .Suppose ¯Wis the weight with context length m, then the
induced Wwhen evaluating on context of length m′is:
W=m
m′¯W
Proof.We note that ¯Wis the un-normalized weight, i.e. scaling with the inverse context size 1/m. Only the
normalized weight is preserved when applying to a sentence with a different context length.
Then, the prediction is given as:
ˆy:=x⊤
qWX⊤y
=x⊤
q1
m′WNormalized X⊤y
=x⊤
q1
m′m¯W
|{z}
=WX⊤y
Thus,
W=m
m′¯W
Lemma 2 (Mixed 4th-Order Moment of Gaussian) .Suppose x∼ N(0, I),r∼ N(0, δ2I), then
1.
E[rr⊤W⊤xx⊤Wrr⊤] = 2δ4W⊤W+δ4tr(W⊤W)I (19)
2.
E[rx⊤W⊤xx⊤Wxr⊤] =
tr 
W2
+ tr
W⊤W
+ tr (W)2
δ2I (20)
3.
E[xr⊤W⊤xx⊤Wrx⊤] = 2δ2WW⊤+δ2tr(W⊤W)I (21)
4.
E[rx⊤W⊤xx⊤Wrx⊤] =δ2
W⊤W+W⊤W⊤+W⊤tr(W)
(22)
5.
E[rr⊤W⊤xx⊤Wxx⊤] =δ2
W⊤W+W⊤W⊤+W⊤tr(W)
(23)
Proof. 1. We have
E
x,r[rr⊤W⊤xx⊤Wrr⊤] =E
r[rr⊤W⊤Wrr⊤]
= 2δ2IW⊤Wδ2I+ tr(W⊤Wδ2I)δ2I
= 2δ4W⊤W+δ4tr(W⊤W)I
= 2δ4W⊤W+δ4tr(W⊤W)I(24)
where the first step follows from Equation (32).
2.
E
x,r[rx⊤W⊤xx⊤Wxr⊤] =E
rh
rE
xh
x⊤W⊤xx⊤Wxi
r⊤i
=
tr 
W2
+ tr
W⊤W
+ tr (W)2
δ2I(25)
where the first step follows from Equation (34).
16

3.
E[xr⊤W⊤xx⊤Wrx⊤] =Eh
xEh
tr
r⊤W⊤xx⊤Wri
x⊤i
=Eh
xEh
tr
rr⊤W⊤xx⊤Wi
x⊤i
=δ2Extr(W⊤xx⊤W)x⊤
=δ2Extr(x⊤WW⊤x)x⊤
=δ2Exx⊤WW⊤xx⊤
= 2δ2WW⊤+δ2tr(WW⊤)I
= 2δ2WW⊤+δ2tr(W⊤W)I(26)
where the first three steps follow from the cyclic property of trace and the last step follows from
Equation (32).
4.
E[rx⊤Wxx⊤Wrx⊤] =E
r(x⊤Wr)⊤x⊤Wxx⊤
=Eh
rr⊤W⊤xx⊤Wxx⊤i
=δ2Eh
W⊤xx⊤Wxx⊤i
=δ2W⊤
W+W⊤+ tr(W)I
=δ2
W⊤W+W⊤W⊤+W⊤tr(W)
where the first step follows from x⊤Wrbeing scalar, and the third step follows from Equation (32).
5.
E[rr⊤W⊤xx⊤Wxx⊤] =δ2W⊤
W+W⊤+ tr (W)
=δ2
W⊤W+W⊤W⊤+W⊤tr(W) (27)
It follows from the application of Equation (32).
Lemma 3 (Expectation of 6th-Order Gaussian Monomial) .Ifx∼ N(0, I), then
E[xx⊤Axx⊤Bxx⊤] =AB+AB⊤+A⊤B+A⊤B⊤+B⊤A+B⊤A⊤+BA+BA⊤
+ tr(B)A+ tr(B)A⊤+ tr(A)B+ tr(A)B⊤
+ tr(A) tr(B)I+ tr 
AB⊤
I+ tr(AB)I(28)
E[xx⊤W⊤xx⊤Wxx⊤] =E[xx⊤Wxx⊤Wxx⊤]
= 2
W2+W⊤W⊤+W⊤W+WW⊤+ tr (W)W+ tr (W)W⊤
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I(29)
Proof.LetT:=E[xx⊤Axx⊤Bxx⊤]. Then, let’s consider one scalar entry:
Tij=E
X
k,ℓ,m,nxixkAkℓxℓxmBmnxnxj
=X
k,ℓ,m,nAkℓBmn·E[xixkxℓxmxnxj] (30)
We now need to compute the 6th-order central moment of standard normal variables. This can be
computed using the Isserlis’ Theorem [Isserlis, 1918]:
17

E[x1···xs] =X
p∈P2sY
(i,j)∈pE[xixj] (31)
where P2
sstands for all distinct ways of partitioning {1, . . . , s }into pairs i, j(perfect matching), and the
product is over the pairs contained in p.
We note that the number of perfect matching for sexamples is given as:
#perfect matching =s!
2s/2(s/2)!
where 2s/2is for ignoring the ordering inside pairs and (s/2)!is for ignoring the ordering between pairs.
We note that there are6!
23·3!= 15distinct partitions for the 6-th order product of Gaussian random
variable. Suppose (xa, xb),(xc, xd),(xe, xf)is a valid pairing, then:
E[xaxb]E[xcxd]E[xexf] =(
1ifa=b, c=d, e=f
0else=δab·δcd·δef
where δij:= 1[i=j]stands for the Kronecker delta.
Here, we will discuss the result for all 15 distinct pairings:
1.(i, k)(l, m)(n, j)X
k,l,m,nAklBmn=X
mAimBmj=Ai·B·j= (AB)ij
2.(i, k)(l, n)(m, j)X
k,l,m,nAklBmn=X
mAimBjm=Ai·Bj·= (AB⊤)ij
3.(i, k)(l, j)(m, n)X
k,l,m,nAklBmn=X
mAijBmm= tr( B)Aij
4.(i, l)(k, m)(n, j)X
k,l,m,nAklBmn=X
mAmiBmj=A·iB·j= (A⊤B)ij
5.(i, l)(k, n)(m, j)X
k,l,m,nAklBmn=X
kAkiBjk=A·iBj·= (A⊤B⊤)i,j
6.(i, l)(k, j)(m, n)X
k,l,m,nAklBmn=X
mAjiBmm= (A⊤)ijtr(B)
7.(i, m)(k, l)(n, j)X
k,l,m,nAklBmn=X
kAkkBij= tr( A)Bij
8.(i, m)(k, n)(l, j)X
k,l,m,nAklBmn=X
kAkjBik=A·jBi·= (BA)ij
9.(i, m)(k, j)(l, n)X
k,l,m,nAklBmn=X
lAjlBil=Aj·Bi·= (BA⊤)ij
18

10.(i, n)(k, l)(m, j)X
k,l,m,nAklBmn=X
kAkkBji= tr( A)(B⊤)ij
11.(i, n)(k, m)(l, j)X
k,l,m,nAklBmn=X
mAmjBmi=A·jB·i= (B⊤A)ij
12.(i, n)(k, j)(l, m)X
k,l,m,nAklBmn=X
mAjmBmi=Aj·B·i= (B⊤A⊤)ij
13.(i, j)(k, l)(m, n)X
k,l,m,nAklBmn=X
k,mAkkBmm= tr( A) tr(B)δij
14.(i, j)(k, m)(l, n)X
k,l,m,nAklBmn=X
k,lAklBkl= tr( AB⊤)δij
15.(i, j)(k, n)(l, m)X
k,l,m,nAklBmn=X
m,kAkmBmk= tr( AB)δij
Summing up all of these 15 terms together, we obtain Eq. (28). Then, we plug in A=W, B=W⊤, we
obtain Eq. (29).
Lemma 4 (4th-Order Gaussian Monomial) .Letx,x1, . . . ,xm∼ N(0, I)andX= [x⊤
1;. . .;x⊤
m]. Then, we
have
Exx⊤Wxx⊤=W+W⊤+ tr(W)I
= 2W+ tr(W)IifWis symmetric(32)
and
EX⊤XWX⊤X=m2W+mW⊤+mtr(W)I
=m(m+ 1)W+mtr(W)IifWis symmetric(33)
Ex⊤Axx⊤Bx= tr 
A 
B+B⊤
+ tr(A) tr(B) (34)
IfA=W⊤, B=W, then
Ex⊤W⊤xx⊤Wx =Ex⊤Wxx⊤Wx = tr(W⊤W) + tr( W2) + tr( W)2(35)
Proof.Equation (32) follows from section 8.2.4 of [Petersen et al., 2008] by plugging in mean 0 and variance
I.
EX⊤XWX⊤X=X
ixix⊤
iWx ix⊤
i+X
i̸=jxix⊤
iWx jx⊤
j
=m
W+W⊤+ tr(W)I
+m(m−1)W
=m2W+mW⊤+mtr(W)I
=m(m+ 1)W+mtr(W)I(36)
where the second step follows from plugging in Equation (32).
Equation (34) follows from section 8.2.4 of [Petersen et al., 2008] by plugging in mean 0 and variance I.
19

B Additional Proof for RAG
Here, we provide an overview of the organization of the proof. First, we consider the uniform retrieval
noise scenario, and compute the population loss for generic Win Theorem 1. Then, we plug in the special
caseW∗(isotropic pretrained weight), and provide a closed-form loss in Proposition 3. Then, we analyze
its finite sample complexity in Proposition 1 and the optimal RAG examples in relation to ICL examples
in Proposition 2.
Later on, we provide an finite sample complexity analysis for non-uniform retrieval noise, Theorem 2 for
Distance Proportional Noise, and Theorem 3 for Distance-Weighted Mixture Noise.
B.1 Uniform Retrieval Noise
Theorem (Restatement of Theorem 1) .Under Assumption 1, 2, 3, the population loss of the linear
self-attention predictor ˆyq=x⊤
qWX⊤ysatisfies
Ltt+rag (W) =E(E(ˆyq)−ˆyq)2
|{z }
:=err variance (W)+E(E(ˆyq)−E(yq))2
| {z }
:=err bias(W)+ σ2
|{z}
irreducible noise(37)
Specifically,
errvariance (W) =
mσ2+ 
1 +δ2
nσ2
rag
tr(W⊤W) +nσ2
ragtr(W2) +nσ2
ragtr(W)2
errbias(W) =β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I+M4i
βtt
=β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I
+
n2 
2 +δ2
+n 
m+δ2
| {z }
:=c1
W2+ 
W2⊤
+ 2n(n+δ2)|{z}
:=c2WW⊤
+
m2+m+mn 
2 + 2 δ2
+n2 
2 + 2 δ2+δ4
+n 
2δ2+δ4
| {z }
:=c3W⊤W
+
n2 
2 +δ2
+n 
m+δ2
| {z }
:=c4, c4=c1
tr(W)
W+W⊤
+
n2+nδ2
|{z}
:=c5 
tr(W)2+ tr 
W2
I+
m+n2+n 
2δ2+δ4
| {z }
:=c6tr
W⊤W
I
βtt(38)
Proof.For computational convenience, I will define the following quantities for Gram matrix: G0=X⊤
iclXicl,
Gi:= (xq+ri)(xq+ri)⊤, and G:=G0+P
i∈[n]Gi.
We write down the error explicitly:
yq−x⊤
qWX⊤y=x⊤
qβtt+ϵq−x⊤
qWX⊤Xβtt−x⊤
qWX⊤ϵ
=x⊤
q(βtt−WGβtt)−x⊤
qWX⊤ϵ+ϵq
=x⊤
q(I−WG)βtt−x⊤
qWX⊤ϵ+ϵq(39)
Therefore, the population loss is equal to:
Ltt+rag (W) = E
(xq,yq),(X,y),ϵ,r
x⊤
q(I−WG)βtt−x⊤
qWX⊤ϵ2
+σ2
We note that both ϵiclandϵragare independent of xq,X,r, andE[ϵ] = 0.
E
ϵh
−2 
x⊤
q(I−WG)βtt
x⊤
qWX⊤ϵi
= 0
20

And therefore, we have the following loss decomposition:
Ltt+rag (W) = E
xq,X,r,ϵ
x⊤
qWX⊤ϵ2
+E
xq,X,rh 
x⊤
q(I−WG)βtt2i
+σ2(40)
Then, we compute the mean of the prediction and the label:
E
ϵqyq=E
ϵq 
x⊤
qβtt+ϵq
=x⊤
qβtt
E
ϵˆyq=Ex⊤
qWX⊤y
=Ex⊤
qWX⊤(Xβtt+ϵ)
=Ex⊤
qWGβtt
And further, we have
E
ϵq
yq−E
ϵqyq2
=E
ϵq 
x⊤
qβtt+ϵq−x⊤
qβtt2=E
ϵqϵ2
q=σ2

ˆyq−E
ϵˆyq2
=
x⊤
qWX⊤(Xβtt+ϵ)−x⊤
qWX⊤Xβtt2
=
x⊤
qWX⊤ϵ2

E
ϵq(yq)−E
ϵˆyq2
= 
x⊤
qβtt−x⊤
qWGβtt2= 
x⊤
q(I−WG)βtt2(41)
If we plug Equation (41) into the loss decomposition Equation (40), we have
Ltt+rag (W) = E
xq,X,r,ϵ
x⊤
qWX⊤ϵ2
+E
xq,X,rh 
x⊤
q(I−WG)βtt2i
+σ2
=E
E
ϵ(ˆyq)−ˆyq2
| {z }
:=err variance (W)+E
E
ϵq(ˆyq)−E
ϵ(yq)2
| {z }
:=err bias(W)+E
yq−E
ϵqyq2
|{z }
=σ2
(irreducible noise)(42)
and we can obtain the bias-variance tradeoff as given in Equation (37).
Compute Exq,X,r,ϵ
x⊤
qWX⊤ϵ2
First, we let
z:=x⊤
qWX⊤ϵ=m+nX
i=1x⊤
qWx i·ϵi
Then,
z2=m+nX
i,j=1(x⊤
qWx i)(x⊤
qWx j)ϵiϵj=m+nX
i,j=1(x⊤
iW⊤xq)(x⊤
qWx j)ϵiϵj
Taking expectation:
E
ϵ[z2] =m+nX
i,j=1(x⊤
iW⊤xq)(x⊤
qWx j)·E[ϵiϵj]
Because the noise terms are independent and zero-mean, we have:
E[ϵiϵj] =

σ2, i =j≤m
σ2
rag, i=j > m
0, i ̸=j
So only the diagonal terms survive:
E[z2] =mX
i=1σ2·E
(x⊤
qWx i)2
+m+nX
i=m+1σ2
rag·E
(x⊤
qW(xq+ri−m))2
21

•ICL Term: Since xq,xi∼ N(0, I)and are independent,
E[(x⊤
qWx i)2] =Eh
x⊤
iW⊤xqx⊤
qWx ii
=Eh
tr
W⊤xqx⊤
qWx ix⊤
ii
= tr(W⊤W)(43)
where the first step follows from the cyclic property of trace, the last step follows from the symmetry of
W.
⇒ICL contribution =m·σ2·tr(W⊤W) (44)
•RAG Term:
Each row in RAG has the form xq+ri, so:
x⊤
qW(xq+ri) =x⊤
qWx q+x⊤
qWr i
Then, we plug in Equation (34) into the RAG term:
Eh 
x⊤
qW(xq+ri)2i
=E[(x⊤
qWx q)2] +E[(x⊤
qWr i)2] + 2E[x⊤
qWx q·x⊤
qWr i]
=E[(x⊤
qWx q)2] +E[(x⊤
qWr i)2]
=E[(x⊤
qWx q)2] +δ2·tr(W⊤W)
=h
tr(W⊤W) + tr( W2) + tr( W)2i
+δ2·tr(W⊤W)
= tr(W⊤W) + tr( W2) +δ2tr(W⊤W) + tr( W)2(45)
where the second step follows from E[ri] = 0, the third step follows from the cyclic property of trace.
⇒RAG contribution =n·σ2
rag·h
(1 +δ2) tr(W⊤W) + tr( W2) + tr( W)2i
Thus, we can combine the two terms above and obtain the following:
E
x⊤
qWX⊤ϵ2
=
mσ2+ 
1 +δ2
nσ2
rag
tr(W⊤W) +nσ2
ragtr(W2) +nσ2
ragtr(W)2(46)
Compute Exq,X,r
(x⊤
q(I−WG)βtt)2
First, we can expand the expectation and decompose the inner
terms into 4 terms:
E
xq,X,rh
(I−WG)⊤xqx⊤
q(I−WG)i
=E
xq,X,r
I−GW⊤
xqx⊤
q(I−WG)
=Exqx⊤
q|{z}
:=M1−Exqx⊤
qWG
|{z}
:=M2−EGW⊤xqx⊤
q|{z }
:=M3+EGW⊤xqx⊤
qWG
| {z }
:=M4(47)
We denote the four pieces M1, M2, M3, M4in order. First, we note that:
M1=E
xqx⊤
q
=I
Then, we expand out the terms in M2:
22

E
xq,rxqx⊤
qWG =
E
xq,rxqx⊤
q
WG 0+E
xq,rxqx⊤
qWnX
i=1(xq+ri)(xq+ri)⊤
=WG 0+E
xq,rxqx⊤
qWnX
i=1(xq+ri)(xq+ri)⊤
=WG 0+E
xq,rxqx⊤
qWnX
i=1(xqx⊤
q+rir⊤
i)
=WG 0+E
xq,rxqx⊤
qWnX
i=1(xqx⊤
q+δ2I)
=WG 0+n(W+W⊤+ tr(W)I) +nδ2W
=WG 0+n(1 +δ2)W+nW⊤+ntr(W)I(48)
where the first step follows from the independence between Xandxq, the second step follows from
Eri= 0,∀i∈[n], the third step follows from the expectation of rir⊤
i=δ2I, and the last step follows from
Equation (32). Then,
M2=WG 0+n(1 +δ2)W+nW⊤+ntr(W)I
= (nδ2+n+m)W+nW⊤+ntr(W)I(49)
Similarly, M3=M⊤
2= (nδ2+n+m)W⊤+nW+ntr(W)I. Now, we perform similar expansion for M4:
M4=E
xq,X,r[GW⊤xqx⊤
qWG]
=E
xq,X,r

G0+X
i∈[n]Gi
W⊤xqx⊤
qW
G0+X
i∈[n]Gi


=E
xq,X,r
G0W⊤xqx⊤
qWG 0+G0W⊤xqx⊤
qWX
i∈[n]Gi+X
i∈[n]GiW⊤xqx⊤
qWG 0
+X
i∈nGiW⊤xqx⊤
qWG i+X
i,j∈n,i̸=jGiW⊤xqx⊤
qWG j

=E
xq,X,r
G0W⊤xqx⊤
qWG 0+nG0W⊤xqx⊤
qWG i| {z }
i∈[n]+nGiW⊤xqx⊤
qWG 0| {z }
i∈[n]
+nGiW⊤xqx⊤
qWG i| {z }
i∈[n]+n(n−1)GiW⊤xqx⊤
qWG j| {z }
i,j∈[n],i̸=j
(50)
First, we can compute that:
M41:=E
xq,X,r[G0W⊤xqx⊤
qWG 0] =E
X,r[G0W⊤WG 0]
=m(m+ 1)W⊤W+mtr(W⊤W)I(51)
where the last line follows from Equation (33) and symmetry of W⊤W. Then, ∀i∈[n], we have:
23

M42:=E
xq,X,rG0W⊤xqx⊤
qWG i=E
xq,X,rG0W⊤xqx⊤
qW(xq+ri) (xq+ri)⊤
=E
xq,X,rG0W⊤xqx⊤
qW 
xqx⊤
q+rir⊤
i
=E
xq,X,rG0W⊤
W+W⊤+ tr(W) +Wδ2
=m
W⊤W+W⊤W⊤+ tr(W)W⊤+δ2W⊤W
=m
(1 +δ2)W⊤W+W⊤W⊤+ tr(W)W⊤(52)
where the first steps follows from E[ri] = 0, the second step follows from Equation (32).
Moreover, we note that ∀i∈[n]:
M43:=E
xq,X,riGiW⊤xqx⊤
qWG i= (xq+ri)(xq+ri)⊤W⊤xqx⊤
qW(xq+ri)(xq+ri)⊤
= (xqx⊤
q+rix⊤
q+xqr⊤
i+rir⊤
i)W⊤xqx⊤
qW(xqx⊤
q+rix⊤
q+xqr⊤
i+rir⊤
i)
=xqx⊤
qW⊤xqx⊤
qWx qx⊤
q| {z }
0 order in ri+rir⊤
iW⊤xqx⊤
qWr ir⊤
i| {z }
4th-order in ri
+ (rix⊤
q+xqr⊤
i)W⊤xqx⊤
qW(rix⊤
q+xqr⊤
i)
| {z }
2nd-order in ri
+rir⊤
iW⊤xqx⊤
qWx qx⊤
q+xqx⊤
qW⊤xqx⊤
qWr ir⊤
i| {z }
2nd-order in ri
(53)
It worth noting that given Gaussian vector ri, then its monomial of odd order has 0 expectation according
to Isserlis’ Theorem [Isserlis, 1918]. And we can thus obtain the third line by keeping only the even order
monomials of ri.
24

By adding up Lemma 3 and all the terms above, we obtain that:
E
xq,X,riGiW⊤xqx⊤
qWG i
=2
W2+ (W2)⊤+W⊤W+WW⊤+ tr (W) (W+W⊤)
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I

0th-order in ri, Lemma 3
+ 2δ4W⊤W+δ4tr(W⊤W)I| {z }
4th-order in ri, Equation (19)
+δ2h
tr (W)
W⊤+W
+W2+ (W2)⊤+ 2W⊤Wi
| {z }
Equation (23) and its transpose
+
tr 
W2
+ tr
W⊤W
+ tr (W)2
δ2I
| {z }
Equation (20)
+ 2δ2WW⊤+δ2tr(W⊤W)I| {z }
Equation (21)
+δ2h
tr (W)
W⊤+W
+W2+ (W2)⊤+ 2W⊤Wi
| {z }
Equation (22) and its transpose
= (2 + 2 δ2)h
tr (W)
W⊤+W
+W2+ (W2)⊤i
+ (2 + 4 δ2)W⊤W+ 2WW⊤
+ (1 + δ2)h
tr (W)2I+ tr 
W2
I+ tr
W⊤W
Ii
+ 2δ4W⊤W+δ4tr(W⊤W)I+ 2δ2WW⊤+δ2tr(W⊤W)I
= (2 + 2 δ2)h
tr (W)
W⊤+W
+W2+ (W2)⊤i
+ (2 + 4 δ2+ 2δ4)W⊤W+ (2 + 2 δ2)WW⊤
+ (1 + δ2)
tr (W)2+ tr 
W2
I+ (1 + 2 δ2+δ4) tr(W⊤W)I(54)
25

Also, we expand the cross-term out for ∀i, j∈[n], i̸=j:
M44:=EGiW⊤xqx⊤
qWG j=E(xq+ri)(xq+ri)⊤W⊤xqx⊤
qW(xq+rj)(xq+rj)⊤
= 
xqx⊤
q+rir⊤
i
W⊤xqx⊤
qW 
xqx⊤
q+rjr⊤
j
=xqx⊤
qW⊤xqx⊤
qWx qx⊤
q+rir⊤
iW⊤xqx⊤
qWx qx⊤
q
+xqx⊤
qW⊤xqx⊤
qWr jr⊤
j+rir⊤
iW⊤xqx⊤
qWr jr⊤
j
= 2
W2+ 
W2⊤+W⊤W+WW⊤+ tr (W)W+ tr (W)W⊤
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I
+δ2
W2+ (W2)⊤+ 2W⊤W
+ tr(W)(W+W⊤) +δ4W⊤W
= (2 + δ2)
W2+ 
W2⊤+ tr (W)W+ tr (W)W⊤
+ (2 + 2 δ2)W⊤W+ 2WW⊤
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I+δ4W⊤W
= (2 + δ2)
W2+ 
W2⊤+ tr (W)W+ tr (W)W⊤
+ (2 + 2 δ2+δ4)W⊤W+ 2WW⊤
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I(55)
where the first step follows from the independence of xq,ri,rj, and the second step follows from applying
Lemma 3 and Equation (32).
Combining the above terms together, we have
M4=M41+n(M42+M⊤
42) +nM43+n(n−1)M44
=m(m+ 1)W⊤W+mtr(W⊤W)I+mn 
2 + 2 δ2
W⊤W+W2+ (W2)⊤+ tr (W)
W+W⊤
+nM43+n(n−1)M44
= 2n 
2n+δ2
W2+ 2n 
n+δ2
WW⊤
+
m2+m+ (4 + 2 δ2)mn+n2 
2 + 4 δ2+δ4
+n 
2δ2+δ4
W⊤W
+
n2 
2 +δ2
+n 
m+δ2
tr(W)
W+W⊤
+ 
n2+nδ2 
tr(W)2+ tr(W2)
I+
m+n2+n 
2δ2+δ4
tr
W⊤W
I
=
n2 
2 +δ2
+n 
m+δ2
W2+ 
W2⊤+ tr(W)
W+W⊤
+
2n2+ 2nδ2
WW⊤
+
m2+m+mn 
2 + 2 δ2
+n 
2δ2+δ4
+n2 
2 + 2 δ2+δ4
W⊤W
+
n2+nδ2 
tr(W)2+ tr 
W2
I
+
m+n2+n 
2δ2+δ4
tr
W⊤W
I
(56)
In summary, combining all terms together, we have:
L(W) := err variance + err bias+σ2
where the irreducible variance isσ2, and the reducible variance (variance of ICL + RAG) is
Variance of ICL +Variance of RAG =
mσ2+ 
1 +δ2
nσ2
rag
tr(W⊤W) +nσ2
ragtr(W2) +nσ2
ragtr(W)2
And the err from the bias term is given as:
26

errbias=β⊤
tt[M1−M2−M3+M4]βtt
=β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I+M4i
βtt
=β⊤
tth
I−(nδ2+ 2n+m)(W+W⊤)−2ntr(W)I
+
n2 
2 +δ2
+n 
m+δ2
W2+ 
W2⊤
+ 2n(n+δ2)WW⊤
+
m2+m+mn 
2 + 2 δ2
+n2 
2 + 2 δ2+δ4
+n 
2δ2+δ4
W⊤W
+
n2 
2 +δ2
+n 
m+δ2
tr(W)
W+W⊤
+
n2+nδ2 
tr(W)2+ tr 
W2
I+
m+n2+n 
2δ2+δ4
tr
W⊤W
Ii
βtt
The previous theorem gives the exact form the RAG population with general W. In the following
proposition, we will compute the population under special Win order to obtain a more fine-grained
complexity analysis.
Proposition 3 (RAG Population loss under isotropic setting) .Assuming W∗=m
(m+d+1)(m+n)I. Then, the
population loss are given as:
Ltt+rag (W∗) = err variance (W∗) + err bias(W∗) +σ2
errvariance (W∗) =m3d
[(m+d+ 1)( m+n)]2σ2+dm2n(2 +δ2+d)
[(m+d+ 1)( m+n)]2σ2
rag
errbias(W∗) =∥βTT∥2
2
1−2m
(m+d+ 1)( m+n) 
nδ2+ 2n+m+nd
+P(m, n, d, δ )m2
(m+d+ 1)2(m+n)2
where
P(m, n, d, δ ) =6n2+ 4nδ2+m2+m+ 
4 + 2 δ2
mn
+n2 
2 + 4 δ2+δ4
+n 
2δ2+δ4
+ 2dn2 
2 +δ2
+ 2dn 
m+δ2
+d(d+ 1) 
n2+nδ2
+dm+dn2+dn 
2δ2+δ4
Proof.Plugging in the value of W∗, we first compute the error from input variance.
tr((W∗)2) =dm2
(m+d+ 1)2(m+n)2
tr(W∗) =dm
(m+d+ 1)( m+n)
errvariance (W∗) =
mσ2+ 
1 +δ2
nσ2
rag
tr
W⊤W
+nσ2
ragtr 
W2
+nσ2
ragtr(W)2
=dm2[mσ2+ (1 + δ2)nσ2
rag]
(m+d+ 1)2(m+n)2+nσ2
ragdm2
(m+d+ 1)2(m+n)2+nσ2
ragd2m2
(m+d+ 1)2(m+n)2
=m3d
[(m+d+ 1)( m+n)]2σ2+dm2n(2 +δ2+d)
[(m+d+ 1)( m+n)]2σ2
rag
Then, we proceed to plug in the value and compute the error from the estimation bias.
errbias(W∗) =∥βtt∥2
2
1−2m(nδ2+ 2n+m)
(m+n)(m+d+ 1)−2ndm
(m+n)(m+d+ 1)+m2
(m+d+ 1)2(m+n)2(. . .)|{z}
P(m,n,d,δ )

=∥βTT∥2
2
1−2m
(m+d+ 1)( m+n) 
nδ2+ 2n+m+nd
+P(m, n, d, δ )m2
(m+d+ 1)2(m+n)2
27

where
P(m, n, d, δ ) = (2 c1+c2+c3) + 2dc4+ (d2+d)c5+dc6
= 2 
n2 
2 +δ2
+n 
m+δ2
+ 2n(n+δ2)
+
m2+m+mn 
2 + 2 δ2
+n2 
2 + 2 δ2+δ4
+n 
2δ2+δ4
+ 2d[n2(2 +δ2) +n(m+δ2)] + ( d2+d)(n2+nδ2) +d[m+n2+n(2δ2+δ4)]
= 6n2+ 4nδ2+m2+m+ 
4 + 2 δ2
mn
+n2 
2 + 4 δ2+δ4
+n 
2δ2+δ4
+ 2dn2 
2 +δ2
+ 2dn 
m+δ2
+d(d+ 1) 
n2+nδ2
+dm+dn2+dn 
2δ2+δ4
B.1.1 Finite Sample Complexity of RAG
Proposition (Restatement of Proposition 1) .Under Assumption 1, 2, 3, if δ2≪1,
Ltt+rag (W∗) =O
σ2+dm
(m+n)2σ2+d2n
(m+n)2σ2
rag
| {z }
errvariance (W∗)+∥βtt∥2
2"
d
m+d2n
m+n2#
| {z }
errbias(W∗)

errvariance (W∗) =

O(d
mσ2+d2
m2σ2
rag) =O 1
m
m→ ∞,nfixed.
O(d
n2σ2+d2
nσ2
rag) =O 1
n
n→ ∞,mfixed
O(d
mσ2+d2
mσ2
rag) =O 1
m
m, n→ ∞,n= Θ( m)(57)
errbias(W∗) =

O 
∥βtt∥2
2d
m
ifm→ ∞,nis fixed
O 
∥βtt∥2
2d2
=C1 ifn→ ∞,mis fixed
O 
∥βtt∥2
2 d
m+d2
=C2+O(∥βtt∥2
2d
m)ifm→ ∞,n= Θ( m)(58)
Proof.We will bound the variance-induced error and the bias-induced error separately.
Variance-Induced Error First, we try to bound errvariance (W∗):
errvariance (W∗) =dm3
(m+d+ 1)2(m+n)2σ2+dm2n(2 +δ2+d)
(m+d+ 1)2(m+n)2σ2
rag
≤dm3
m2(m+n)2σ2+dm2n(d+δ2+ 2)
m2(m+n)2σ2
rag
=dm
(m+n)2σ2+d(2 +δ2+d)n
(m+n)2σ2
rag
=Odm
(m+n)2σ2+d2n
(m+n)2σ2
rag
=

O(d
mσ2+d2
m2σ2
rag) =O 1
m
m→ ∞,nfixed.
O(d
n2σ2+d2
nσ2
rag) =O 1
n
n→ ∞,mfixed
O(d
mσ2+d2
mσ2
rag) =O 1
m
m, n→ ∞,n= Θ( m)(59)
where the second line follows from (m+d+ 1)≥mand the fourth line follows from the fact that δ2is small
relative to m, n, d.
28

Bias-Induced Error We will expand out the term
errbias(W∗) =∥βtt∥2
2Q(m, n;d, δ2)
(m+d+ 1)2(m+n)2(60)
where
Q(m, n;d, δ2) := ( m+n)2(m+d+ 1)2−2m(m+n)(m+d+ 1)( nδ2+ 2n+m+nd) +m2P(m, n, d, δ2)
= (d+ 1)m3+ (d2+ 2dδ2+ 4d+δ4+ 2δ2+ 5)| {z }
:=κ22m2n2
+ (d2δ2−2d2+dδ4+ 3dδ2−4d+δ4+ 4δ2−2)| {z }
:=κ21m2n
− 
2d2+ 2dδ2+ 4d+ 2δ2+ 2
| {z }
:=κ12mn2+ (d2+ 2d+ 1)( m+n)2
= (d+ 1)m3+κ22m2n2+|κ21|m2n+lower-order terms
≤(d+ 1)m3+κ22m2n2+|κ21|m2n+ (d+ 1)2(m+n)2
(61)
where the last line follows from κ12<0.
Note that we assume δ2≪1. Now, we can bound each of the term in Qdivided individually:
•Cubic term:
(d+ 1)m3
m2(m+n)2=d+ 1
mm
m+n2
≤d+ 1
m(62)
•Skew-cubic term:
|κ21|m2n
m2(m+n)2=|κ21|n
(m+n)2≤ |κ21|n
(m+n)2(63)
•Quartic term:
κ22m2n2
m2(m+n)2=κ22n
m+n2
(64)
•last term:
(d+ 1)2(m+n)2 1
m2(m+n)2=d2
m2
Combining Equation (61), Equation (62), Equation (63), Equation (64), we can obtain that
errbias(W∗) =O
∥βtt∥2
2dm
(m+n)2+d2n2
(m+n)2+d2
m2
=

O 
∥βtt∥2
2d
m
ifm→ ∞,nis fixed
O 
∥βtt∥2
2d2
=C1 ifn→ ∞,mis fixed
O 
∥βtt∥2
2 d
m+d2
=C2+O(∥βtt∥2
2d
m)ifm→ ∞,n= Θ( m)(65)
where the third step follows from plugging in the highest order monomial of dfrom κ21, κ22.
B.1.2 Optimality of Number of RAG Examples
Proposition (Restatement of Proposition 2) .Under Under Assumption 1,2,3, δ2≪1, and reasonable choice
ofσ2, σ2
rag(σ2, σ2
rag≪ ∥βtt∥2
2), the optimal n∗that minimizes the RAG loss follows:
n∗=O 
m 
d2∥βtt∥2
2+dσ2−d2σ2
rag
md2∥βtt∥2
2−d2σ2rag!
=O 
d∥βtt∥2
2+σ2−dσ2
rag
d∥βtt∥2
2!
(66)
29

and the improvement on loss from picking the optimal n∗over n= 0is given as:
Ltt+rag (W∗)|n=0− Ltt+rag (W∗)|n=n∗=O1
m2
(67)
Proof.First, we define several constants that can lead to a cleaner calculation. Let ω1:=d,ω2:=d2. Then,
errvariance (W∗) =dm3
(m+d+ 1)2(m+n)2σ2+dm2n(2 +δ2+d)
(m+d+ 1)2(m+n)2σ2
rag
≈m3
(m+d+ 1)2(m+n)2ω1σ2+m2n
(m+d+ 1)2(m+n)2ω2σ2
rag
where the last line follows from δ2≪1. Let Q(m, n, d, δ2) :=errbias(W∗)(m+d+1)2(m+n)2
∥βtt∥2
2as in Equation (61).
Then,
Q(m, n;d, δ2) = (m+n)2(m+d+ 1)2−2m(m+n)(m+d+ 1)( nδ2+ 2n+m+nd) +m2P(m, n, d, δ2)
= (d+ 1)m3+ (d2+ 2dδ2+ 4d+δ4+ 2δ2+ 5)m2n2
+ (d2δ2−2d2+dδ4+ 3dδ2−4d+δ4+ 4δ2−2)m2n
− 
2d2+ 2dδ2+ 4d+ 2δ2+ 2
mn2+ (d2+ 2d+ 1)( m2+n2)
≈d|{z}
:=τ30m3+d2
|{z}
τ22m2n2−2d2
|{z}
:=τ21m2n−2d2
|{z}
:=τ12mn2+d2
|{z}
:=τ2(m2+n2)
=τ30m3+τ22m2n2+τ21m2n+τ12mn2+τ2(m2+n2)
(68)
Now, we want to find the optimal n∗w.r.t. Ltt+rag. That is, we want to minimize

m3ω1σ2+m2nω2σ2
rag+∥βtt∥2
2 
τ30m3+τ22m2n2+τ21m2n+τ12mn2+τ2 
m2+n2 1
(m+n)2(m+d+ 1)2
(69)
where all τ, ωare positive except that τ12is negative. First, we take out the terms that does not depend on
n, and we equivalently minimize
L(n) :=
m3ω1σ2+m2nω2σ2
rag+∥βtt∥2
2 
τ30m3+τ22m2n2+τ21m2n+τ12mn2+τ2 
m2+n2 1
(m+n)2
Let
A=m3ω1σ2+∥βtt∥2τ30m3+∥βtt∥2τ2m2,
B=m2 
ω2σ2
rag+∥βtt∥2τ21
,
C=∥βtt∥2 
τ22m2+τ12m+τ2
.(70)
Then,
L(n) = (A+Bn+Cn2)/(m+n)2
Then, by the rule for derivative of quotient,
∂(Ltt+rag (W∗))
∂n=(B+ 2Cn)(m+n)2−2(m+n) 
A+Bn+Cn2
(m+n)4
=(B+ 2Cn)(m+n)−2 
A+Bn+Cn2
(m+n)3
=Bm+Bn+ 2Cmn + 2Cn2−2A−2Bn−2Cn2
(m+n)3
=Bm−Bn+ 2Cmn−2A
(m+n)3
Set the derivative to be zero, we have
30

Bm−Bn+ 2Cmn−2A= 0
and
n⋆=Bm−2A
B−2Cm
=m(m2 
ω2σ2
rag+∥βtt∥2τ21
)−2(m3ω1σ2+∥βtt∥2τ30m3+∥βtt∥2τ2m2)
(m2 
ω2σ2rag+∥βtt∥2τ21
)−2m(∥βtt∥2(τ22m2+τ12m+τ2))
=m 
2∥βtt∥2
2dm+ 2∥βtt∥2
2d+ 2∥βtt∥2
2m−dmσ2
rag+ 2mσ2
d 
2∥βtt∥2
2m2−2∥βtt∥2
2m+ 2∥βtt∥2
2−mσ2rag
≤md 
2∥βtt∥2
2dm−dmσ2
rag+ 2mσ2
d2 
2∥βtt∥2
2m2−2∥βtt∥2
2m+ 2∥βtt∥2
2−mσ2rag
=O 
md 
2∥βtt∥2
2dm−dmσ2
rag+ 2mσ2
d2 
2∥βtt∥2
2m2−mσ2rag!
=O 
m 
d2∥βtt∥2
2+dσ2−d2σ2
rag
md2∥βtt∥2
2−d2σ2rag!
=O 
d∥βtt∥2
2+σ2−dσ2
rag
d∥βtt∥2
2!
where the third step follows from upper bounding the numerator, and the fourth step follows from lower
bounding the denominator.
n∗as Global Minimizer Now, we will show that the stationary point is the global minimizer. The second
order derivative is give as:
∂(Ltt+rag (W∗))
∂n=2 
Cm2−2Cmn−2Bm+Bn+ 3A
(m+n)4(71)
Plug in Bm−Bn∗+ 2Cmn∗−2A= 0, we have
∂(Ltt+rag (W∗))
∂n|n=n∗=2 
Cm2−A
(m−n)(m+n)3≥0 (72)
Since n∗=O(1), we have m > n∗for large m. Also, we have Cm2> Afor large m, thus we have
∂(Ltt+rag (W∗))
∂n|n=n∗≥0, and n∗is the local minimum. Now, we check the first order derivative of n≥n∗,
Bm−Bn+ 2Cmn−2A=Bm−Bn+ 2Cmn−2A−(Bm−Bn∗+ 2Cmn∗−2A)
=−B(n−n∗) + 2Cm(n−n∗)≥0
where it follows from B≤0, C≥0. Similarly, we can show that Bm−Bn+ 2Cmn−2A≤0,∀n≤n∗.
Thus, we have n∗to be the global minimum of the loss.
Improvement from n∗Here, we plug in n=n∗andn= 0into Equation (69). We have
Ltt+rag|n=n∗(W∗) =A+Bn∗+C(n∗)2
(m+n∗)2(m+d+ 1)2
Ltt+rag|n=0(W∗) =A
m2(m+d+ 1)2(73)
31

Then, the improvement is give as
Ltt+rag|n=0(W∗)− L tt+rag|n=n∗(W∗) =A(m+n∗)2−m2(A+Bn∗+C(n∗)2)
m2(m+n∗)2(m+d+ 1)2
=(n∗)2(2Cm−B)
2m2(m+n∗) (m+d+ 1)2
=OCm
m5
=Om2d2∥βtt∥2
2m
m5
=O1
m2(74)
where the second step step follows from Bm−Bn∗+ 2Cmn∗−2A= 0and the third step follows from
n∗=O(1), and the four step follows from B≤0and|B|=O(C). It finishes the proof.
B.2 Non-Uniform Retrieval Noise
Now, we proceed to the proof for non-uniform retrieval noise.
B.2.1 Distance-Proportional Noise
Theorem (Restatement of Theorem 2) .Under Assumption 1, 2, 4, the population loss is given as:
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1γ1δ2
i[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
If the variance of the retrieval distance follows power law, i.e. ∃γ2>0, q≥0s.t.δ2
i=γ2iq, then
ˆ errbias(W∗) =O
errbias(W∗) +∥βtt∥2
2dn2q+1+n2q+2
(m+n)2
(75)
and
ˆ errvariance (W∗) =Odmσ2+d(n2q+1)σ2
(m+n)2
=(
O 
dn2q−1σ2
ifn→ ∞,q≤1/2
diverges if n→ ∞,q >1/2(76)
Proof.We first write down the error explicitly similar to Equation (39).
yq−x⊤
qWX⊤y=x⊤
q(I−WG)βtt−x⊤
qWX⊤ϵ+ϵq
And we can break down the population loss as
ˆLtt+rag (W) = E
(xq,yq),(X,y),ϵ,r 
x⊤
q(I−WG)βtt2+
x⊤
qWX⊤ϵ2
+σ2(77)
Variance-Induced Error
ˆ errvariance (W) =E(x⊤
qWX⊤ϵ)2
=m+nX
i,j=1(x⊤
iW⊤xq)(x⊤
qWx j)E(ϵiϵj)(78)
Because the noise are independent and zero-mean, we have
E[ϵjϵj] =

σ2, i =j≤m
σ2
rag,i, i=j > m
0, i ̸=j
32

Then,
LHS =mX
i=1σ2E[(x⊤
qWx i)2] +m+nX
i=m+1σ2
rag,i−m·E[x⊤
qW(xq+ri−m)2]
Thus, the ICL contribution remains the same as Theorem 1, i.e.
mX
i=1σ2E[(x⊤
qWx i)2] =mσ2tr(W⊤W)
To compute the RAG contribution, we evaluate the formula similar to Equation (45).
Eh 
x⊤
qW(xq+ri)2i
=E[(x⊤
qWx q)2] +E[(x⊤
qWr i)2] + 2E[x⊤
qWx q·x⊤
qWr i]
= tr(W⊤W) + tr( W2) +δ2
itr(W⊤W) + tr( W)2(79)
And thus, the RAG error contribution is
m+nX
i=m+1σ2
rag,i−m·E[x⊤
qW(xq+ri−m)2] =nX
i=1σ2
rag,i[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
Plug in σ2
rag,i=γ1δ2
i, and combining all terms together, we have
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1γ1δ2
i[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
Now, if we further assume δ2
i=γ2iq, and plug in the value of
ˆ errvariance (W∗) =mσ2tr 
(W∗)⊤W∗
+nX
i=1γ1γ2iq[(1 + γ2iq) tr 
(W∗)⊤W∗
+ tr 
(W∗)2
+ tr(W∗)2]
=m2
(m+d+ 1)2(m+n)2"
dmσ2+γ1γ2"
 
2d+d2nX
i=1iqσ2+dγ2nX
i=1i2qσ2##
=m2
(m+d+ 1)2(m+n)2
dmσ2+γ1γ2σ2 
2d+d2
Onq+1
q+ 1+nq
2
+dγ2On2q+1
2q+ 1+n2q
2
=Odmσ2+d(n2q+1)
(m+n)2
=

O 
dn2q−1σ2
ifn→ ∞,q <1/2
O 
dσ2
ifn→ ∞,q= 1/2
diverges if n→ ∞,q >1/2
where the second step follows from the Euler–Maclaurin expansion of the power sum.
Bias-Induced Error From Equation (56), we note that
errbias(W) =β⊤
tt
M1−M2−M3+M41+nX
i=1(M42+M⊤
42) +nX
i=1M43+X
i̸=j,i,j∈[n]M44
βtt
Specifically,
E
xq,X,rh
(I−WG)⊤xqx⊤
q(I−WG)i
=E
xq,X,r
I−GW⊤
xqx⊤
q(I−WG)
=Exqx⊤
q|{z}
:=M1−Exqx⊤
qWG
|{z}
:=M2−EGW⊤xqx⊤
q|{z }
:=M3+EGW⊤xqx⊤
qWG
| {z }
:=M4(80)
33

To avoid the repeated computation, we will highlight the calculation that involves δi, omit some calculation
steps given in the standard case and discuss its bound after allowing for non-uniform offset. We will only
compute δ2
i-involving term and use . . .to denote the rest terms, since we assume δ2≪1in proving Theorem 1.
The final bound will be given as
ˆ errbias(W∗) = err bias(W∗) +δ2-involved terms
M1=E
xqx⊤
q
=Iand remains the same. Let sδ:=P
iδ2
i, Sδ:=P
i(δ2
i)2.
Then, we expand out the terms in M2:
M2=E
xq,rxqx⊤
qWG =
E
xq,rxqx⊤
q
WG 0+E
xq,rxqx⊤
qWnX
i=1(xq+ri)(xq+ri)⊤
=WG 0+E
xq,rxqx⊤
qWnX
i=1(xqx⊤
q+rir⊤
i)
=WG 0+E
xq,rxqx⊤
qWnX
i=1(xqx⊤
q+δ2
iI)
=···+sδW(81)
Similarly, M3=M⊤
2=···+sδW⊤. Now, we perform similar expansion for M4.
First, we note that M41=Exq,X,r[G0W⊤xqx⊤
qWG 0]is independent of δ2
i.
X
i∈[n]M42:=X
i∈[n]E
xq,X,rG0W⊤xqx⊤
qWG i
=X
i∈[n]E
xq,X,rG0W⊤xqx⊤
qW(xq+ri) (xq+ri)⊤
=X
i∈[n]E
xq,X,rG0W⊤xqx⊤
qW 
xqx⊤
q+rir⊤
i
=X
i∈[n]E
xq,X,rG0W⊤
W+W⊤+ tr(W) +Wδ2
=X
i∈[n]m
W⊤W+W⊤W⊤+ tr(W)W⊤+δ2W⊤W
=···+msδW⊤W(82)
34

Following the derivation of the 6th-order and 4th-order moments as in Lemma 3 and Lemma 2, we have
X
i∈[n]M43:=X
i∈[n]E
xq,X,riGiW⊤xqx⊤
qWG i
=2
W2+ (W2)⊤+W⊤W+WW⊤+ tr (W) (W+W⊤)
+ tr (W)2I+ tr 
W2
I+ tr
W⊤W
I

0th-order in ri, Lemma 3
+ 2δ4
iW⊤W+δ4
itr(W⊤W)I| {z }
4th-order in ri, Equation (19)
+δ2
ih
tr (W)
W⊤+W
+W2+ (W2)⊤+ 2W⊤Wi
| {z }
Equation (23) and its transpose
+
tr 
W2
+ tr
W⊤W
+ tr (W)2
δ2
iI
| {z }
Equation (20)
+ 2δ2
iWW⊤+δ2
itr(W⊤W)I| {z }
Equation (21)
+δ2
ih
tr (W)
W⊤+W
+W2+ (W2)⊤+ 2W⊤Wi
| {z }
Equation (22) and its transpose
=X
i∈[n](2 + 2 δ2
i)h
tr (W)
W⊤+W
+W2+ (W2)⊤i
+X
i∈[n](2 + 4 δ2
i)W⊤W+X
i∈[n]2WW⊤
+X
i∈[n](1 +δ2
i)h
tr (W)2I+ tr 
W2
I+ tr
W⊤W
Ii
+X
i∈[n]
2δ4
iW⊤W+δ4
itr(W⊤W)I+ 2δ2
iWW⊤+δ2
itr(W⊤W)I
=···+ 2sδh
tr (W)
W⊤+W
+W2+ (W2)⊤i
+ (4sδ+ 2Sδ)W⊤W+ 2sδWW⊤
+sδ
tr (W)2+ tr 
W2
I+ (2sδ+Sδ) tr(W⊤W)I(83)
Also, we expand the cross-term out for ∀i, j∈[n], i̸=j:
X
i̸=jM44:=X
i̸=jEGiW⊤xqx⊤
qWG j
=X
i̸=j
xqx⊤
qW⊤xqx⊤
qWx qx⊤
q+rir⊤
iW⊤xqx⊤
qWx qx⊤
q
+X
i̸=j
xqx⊤
qW⊤xqx⊤
qWr jr⊤
j+rir⊤
iW⊤xqx⊤
qWr jr⊤
j
=···+X
i̸=jδ2
i
W2+W⊤W+ tr (W)W
+X
i̸=jδ2
i 
W2⊤+W⊤W+ tr (W)W⊤
+X
i̸=jδ2
iδ2
jW⊤W(84)
35

In the non-uniform noise scenario, 4th-order term in δiwill dominate the 2nd-order term in δi. Thus, we will
plug δ2
i=γ2iq,W∗=m
(m+d+1)(m+n)intoerrbias:
errbias(W∗) = err bias(W∗) +O
β⊤
tt
2nX
i(δ2
i)2(W∗)⊤W∗+X
i̸=j,i∈[n],j∈[n]δ2
iδ2
j(W∗)⊤W∗+nX
i(δ2
i)2tr((W∗)⊤W)I
βtt

= err bias(W∗) +O 
β⊤
tt
dn2q+1(W∗)⊤W∗+n2q+2(W∗)⊤W∗
βtt
= err bias(W∗) +O
β⊤
ttdn2q+1+n2q+2
(m+n)2
βtt
It finishes the proof.
B.2.2 Distance-Weighted Probabilistic Noise
Theorem (Restatement of Theorem 3) .Under Assumption 1, 2, 5, then ˜ errbias(W) = ˆ err bias(W), and
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1 
piσ2
s+ (1−pi)σ2
l
[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
If the variance of the retrieval distance follows power law, i.e. ∃γ2>0, q≥0s.t.δ2
i=γ2iq, then:
˜ errvariance (W∗) =(
O 
cldnq−1σ2−(cl−cs)σ2dnq−1−q˜q
ifn→ ∞,q≤1
diverges if n→ ∞,q >1(85)
Proof.First, we note that ˜ errbias(W) =ˆ errbias(W), since both are independent of σ2
ragand depend on the
same set of ∀i, δ2
i.
We write down error explicitly similar to Equation (39) and break down the population loss as:
˜Ltt+rag (W) = E
(xq,yq),(X,y),ϵ,r 
x⊤
q(I−WG)βtt2+
x⊤
qWX⊤ϵ2
+σ2(86)
We note that ˜ errbias(W) =errbias(W), since the error from bias does not depend on the sample complexity.
˜ errvariance (W) =E(x⊤
qWX⊤ϵ)2
=m+nX
i,j=1(x⊤
iW⊤xq)(x⊤
qWx j)E(ϵiϵj)(87)
Because the noise are independent and zero-mean, we have
E[ϵjϵj] =

σ2, i=j≤m
σ2
s, i=j > m,w.p. p
σ2
l, i=j > m,w.p. 1−p
0, i̸=j
Thus, the ICL contribution remains the same as Theorem 1, i.e.
mX
i=1σ2E[(x⊤
qWx i)2] =mσ2tr(W⊤W)
To compute the RAG contribution, we evaluate the formula similar to Equation (45).
Eh 
x⊤
qW(xq+ri)2i
=E[(x⊤
qWx q)2] +E[(x⊤
qWr i)2] + 2E[x⊤
qWx q·x⊤
qWr i]
= tr(W⊤W) + tr( W2) +δ2
itr(W⊤W) + tr( W)2(88)
36

And thus, the RAG error contribution is
nX
i=1 
piσ2
s+ (1−pi)σ2
l
[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
Plug in σ2
rag,i=γ1δ2
i, and combining all terms together, we have
ˆ errvariance (W) =mσ2tr(W⊤W) +nX
i=1 
piσ2
s+ (1−pi)σ2
l
[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
Now we further assume pi= (1 + δ2
i)−˜q,˜q≥0, and plug in the value of W∗. Let B:=m2
(m+d+1)2(m+n)2,
˜ errvariance (W∗) =mσ2tr(W⊤W) +nX
i=1 
piσ2
s+ (1−pi)σ2
l
[(1 + δ2
i) tr(W⊤W) + tr( W2) + tr( W)2]
=B"
dmσ2+nX
i=1 
clσ2−(1 +δ2
i)−˜q(cl−cs)σ2
(1 +δ2
i)·d+d+d2#
≈B"
dmσ2+clσ2nX
i=1 
dδ2
i+d2
−(cl−cs)σ2nX
i=1 
d(1 +δ2
i)1−˜q+d2(1 +δ2
i)−˜q#
≈B"
dmσ2+clσ2nX
i=1dδ2
i−(cl−cs)σ2nX
i=1d(1 +δ2
i)1−˜q#
≈(
B
dmσ2+clσ2dnq+1−(cl−cs)σ2dlog(n)
if˜q= 1 + 1 /q
B
dmσ2+clσ2dnq+1−(cl−cs)σ2dn1+q−q˜q
else
where the second line follows from omitting the lower order term.
If˜q= 1 + 1 /q, we note that the middle term will dominate the error. And combining all cases, we could
obtain
˜ errvariance (W∗) =

O 
cldnq−1σ2−(cl−cs)σ2dnq−1−q˜q
ifn→ ∞,q≤1
diverges if n→ ∞,q >1
O
cldnq−1σ2+ (cl−cs)d2logn
n2σ2
ifn→ ∞,˜q= 1 + 1 /q
37