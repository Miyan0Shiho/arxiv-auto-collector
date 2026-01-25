# A Note on k-NN Gating in RAG

**Authors**: Gérard Biau, Claire Boyer

**Published**: 2026-01-20 09:01:04

**PDF URL**: [https://arxiv.org/pdf/2601.13744v1](https://arxiv.org/pdf/2601.13744v1)

## Abstract
We develop a statistical proxy framework for retrieval-augmented generation (RAG), designed to formalize how a language model (LM) should balance its own predictions with retrieved evidence. For each query x, the system combines a frozen base model q0 ($\times$ x) with a k-nearest neighbor retriever r (k ) ($\times$ x) through a measurable gate k(x). A retrieval-trust weight wfact (x) quantifies the geometric reliability of the retrieved neighborhood and penalizes retrieval in low-trust regions. We derive the Bayes-optimal per-query gate and analyze its effect on a discordance-based hallucination criterion that captures disagreements between LM predictions and retrieved evidence. We further show that this discordance admits a deterministic asymptotic limit governed solely by the structural agreement (or disagreement) between the Bayes rule and the LM. To account for distribution mismatch between queries and memory, we introduce a hybrid geometric-semantic model combining covariate deformation and label corruption. Overall, this note provides a principled statistical foundation for factuality-oriented RAG systems.

## Full Text


<!-- PDF content starts -->

arXiv:2601.13744v1  [math.ST]  20 Jan 2026A Note on /u1D48C-NN Gating in RAG
B/y.pc G´ /e.pc/r.pc/a.pc/r.pc/d.pc BIAU
Sorbonne Universit ´e, Institut universitaire de France
gerard.biau@sorbonne-universite.fr
/a.pc/n.pc/d.pc C/l.pc/a.pc/i.pc/r.pc/e.pc BOYER
Universit ´e Paris-Saclay, Institut Universitaire de France
claire.boyer@universite-paris-saclay.fr
S/u.pc/m.pc/m.pc/a.pc/r.pc/y.pc
We develop a statistical proxy framework for retrieval-aug mented generation (RAG), designed
to formalize how a language model (LM) should balance its own predictions with retrieved
evidence. For each query /u1D465, the system combines a frozen base model /u1D45E0(· |/u1D465)with a/u1D458-nearest
neighbor retriever ˆ /u1D45F(/u1D458)(· |/u1D465)through a measurable gate /u1D706(/u1D465). A retrieval-trust weight /u1D464fact(/u1D465)
quantiﬁes the geometric reliability of the retrieved neigh borhood and penalizes retrieval in low-
trust regions. We derive the Bayes-optimal per-query gate a nd analyze its eﬀect on a discordance-
based hallucination criterion that captures disagreement s between LM predictions and retrieved
evidence. We further show that this discordance admits a det erministic asymptotic limit governed
solely by the structural agreement (or disagreement) betwe en the Bayes rule and the LM. To
account for distribution mismatch between queries and memo ry, we introduce a hybrid geometric-
semantic model combining covariate deformation and label c orruption. Overall, this note provides
a principled statistical foundation for factuality-orien ted RAG systems.
Some key words : Adaptive gating; Hallucination control; Nearest neighbo rs; Retrieval-augmented generation; Statis-
tical learning.
1. I/n.pc/t.pc/r.pc/o.pc/d.pc/u.pc/c.pc/t.pc/i.pc/o.pc/n.pc
Modern language models (LMs) are impressively ﬂuent and ver satile, yet they can hal-
lucinate, producing outputs that sound convincing but are f actually wrong [ Ji et al. ,2023,
Kalai and Vempala ,2024]. Retrieval-augmented generation (RAG) mitigates this is sue by en-
riching model predictions with information drawn from an ex ternal memory, such as curated
documents, code repositories, or previously answered quer ies. These sources are intended to
provide relevant and potentially reliable contextual evid ence [ Lewis et al. ,2020]. At inference
time, the system retrieves items close to the query in an embe dding space and conditions its
response on them. While eﬀective in improving accuracy and g rounding, this mechanism raises
a central design question: for each query, how much should th e system rely on the LM vs. the
retrieved evidence?
Most current RAG systems address this balance through heuri stics. A common approach con-
catenates the top- /u1D458retrieved items into the prompt [ Lewis et al. ,2020], while others interpolate
model and retrieval outputs using a ﬁxed mixture weight, as i n cache-based or /u1D458NN language
models [ Grave et al. ,2017,Khandelwal et al. ,2020,Xu et al. ,2023]. Although often eﬀective,
these strategies lack adaptive control: when retrieved nei ghbors are noisy or oﬀ-topic, excessive

reliance on retrieval may degrade accuracy, while dominant reliance on the LM can lead to
underuse of factual evidence and increased hallucination r isk. A principled, query-dependent
mechanism for balancing the two sources of information is th erefore needed.
This note develops a simple and mathematically explicit fra mework for studying this balancing
problem. In our model, the system consists of a frozen base pr edictor representing the LM,
a/u1D458-nearest neighbor retriever built from an external memory o f labeled examples, and a gate
that mixes the two. To make the gating decision interpretabl e and data-driven, we introduce a
retrieval-trust weight that quantiﬁes how well the retrieved neighborhood geometr ically supports
the query. This weight acts as a penalty in training, discour aging retrieval when local evidence is
unreliable and yielding an adaptive, geometry-aware gatin g rule amenable to statistical analysis.
A central theme of the analysis is hallucination control. We introduce a quantitative discordance
criterion that captures disagreements between LM predicti ons and retrieved evidence, and show
that the resulting gating rule controls this discordance by activating retrieval only where the
local evidence appears reliable. In this way, our results ma ke explicit how the geometry of the
retrieved neighborhood and the gating mechanism jointly de termine when retrieval improves
factual reliability.
To understand retrieval reliability at scale, we analyze th e asymptotic regime in which both the
memory size /u1D45Band the number of neighbors /u1D458grow with /u1D458//u1D45B→0. In this setting we establish
consistency for the /u1D458-NN retrieval estimator and derive limits for the local disc ordance signal that
governs hallucination variation. These results show that, asymptotically, disagreements between
retrieval and the LM reﬂect genuine structural diﬀerences r ather than ﬁnite-sample noise. We
further consider the eﬀect of a mismatch between the query di stribution and the retrieval memory.
We emphasize that the model we study is not meant to mirror the architectural details of modern
RAG systems, which typically rely on prompt construction, a ttention over retrieved documents, or
other integration mechanisms speciﬁc to large LMs. Rather, our goal is to provide an analytically
tractable proxy that captures the essential statistical fo rces governing retrieval-based correction,
with the aim of clarifying these mechanisms and stimulating further theoretical work on the
foundations of retrieval-augmented generation.
Related work. Our approach connects three lines of research. First, RAG ar chitectures have been
extensively explored in NLP and information retrieval [ Lewis et al. ,2020,Borgeaud et al. ,2022,
Izacard et al. ,2023,Shi et al. ,2024]. Second, the retrieval component is rooted in classical ne arest
neighbor theory [ Gy¨orﬁ et al. ,2006,Biau and Devroye ,2015], providing a rigorous statistical
basis for our analysis. Third, the gating mechanism is relat ed to mixture-of-experts models
[Jacobs et al. ,1991,Shazeer et al. ,2017,Fedus et al. ,2022], although here it is explicitly guided
by geometric trust rather than latent specialization. Prev ious approaches to hallucination control
include self-consistency checks [ Manakul et al. ,2023], factuality-oriented evaluation [ Min et al. ,
2023], and LM calibration [ Ulmer et al. ,2024], but they lack a clear statistical interpretation. The
present note provides such an interpretation by linking ret rieval geometry, probabilistic gating,
and factual reliability within a uniﬁed theoretical framew ork.
2. M/o.pc/d.pc/e.pc/l.pc /s.pc/e.pc/t.pc/u.pc/p.pc
We consider an input-output pair (/u1D44B,/u1D44C)drawn from an unknown distribution on R/u1D451×/u1D4B4,
where/u1D4B4={1,...,/u1D436}is a ﬁnite label set. An external memory
ℳ/u1D45B={(/u1D448/u1D456,/u1D449/u1D456)}/u1D45B
/u1D456=1,(/u1D448/u1D456,/u1D449/u1D456) ∼/u1D444/u1D448/u1D449,

stores i.i.d. reference pairs with /u1D448∈R/u1D451and/u1D449∈/u1D4B4. Throughout the main analysis we assume that
the memory is drawn from the same distribution as (/u1D44B,/u1D44C)—the aligned setting , in which /u1D443/u1D44B=/u1D444/u1D448
and/u1D443/u1D44C|/u1D44B=/u1D444/u1D449|/u1D448. In practice, however, the memory may come from a related but distinct source,
such as previously answered queries or labeled documents. F or simplicity, we focus on the aligned
setting in the main text and return to this more general situa tion in Appendix B. Retrieval operates
in the feature space R/u1D451equipped with a norm /bardbl · /bardbl. For any query /u1D465∈R/u1D451, let/u1D448(1)(/u1D465),...,/u1D448 (/u1D458)(/u1D465)
denote its /u1D458nearest neighbors among {/u1D448/u1D456}/u1D45B
/u1D456=1, with corresponding labels /u1D449(1)(/u1D465),...,/u1D449 (/u1D458)(/u1D465).
(Ties are broken deterministically by index order.)
Base predictor and retriever distribution. The base language model (LM) is frozen throughout
and provides a conditional probability distribution /u1D45E0(· |/u1D465)on/u1D4B4, interpreted as an estimate
of the law of /u1D44Cgiven/u1D44B=/u1D465. On the retrieval side, the external memory induces its own c on-
ditional structure: for any query /u1D465, the retriever constructs a local, nonparametric estimate of
the distribution of /u1D449given/u1D448=/u1D465by averaging the labels of the /u1D458nearest neighbors of /u1D465in the
memory:
ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)=1
/u1D458/u1D458/summationdisplay.1
/u1D457=11{/u1D449(/u1D457)(/u1D465)=/u1D466}, /u1D466∈/u1D4B4.
We write ˆ /u1D45F(/u1D458)(/u1D466|/u1D465)=ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)for/u1D466∈/u1D4B4, and refer to ˆ /u1D45F(/u1D458)(· |/u1D465)as the retriever distribution ,
which approximates the conditional distribution /u1D444/u1D449|/u1D448(· |/u1D465)in the neighborhood of /u1D465. In the
aligned setting considered here, where /u1D443/u1D44B=/u1D444/u1D448and/u1D443/u1D44C|/u1D44B=/u1D444/u1D449|/u1D448, ˆ/u1D45F(/u1D458)(· |/u1D465)coincides with the
target conditional distribution /u1D443/u1D44C|/u1D44B(· |/u1D465).
Retrieval-trust weight. To quantify how well the retrieved neighborhood reﬂects the query, we
deﬁne the retrieval-trust weight
/u1D464fact(/u1D465)=1
/u1D458/u1D458/summationdisplay.1
/u1D457=1exp/parenleftbig− /bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardbl2/parenrightbig. (1)
This scalar measures geometric ﬁdelity between /u1D465and its retrieved neighbors. When the neigh-
borhood is compact, the distances /bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardblare small, the exponential terms are close to
one, and/u1D464fact(/u1D465) ≈1, indicating high trust in retrieval. When neighbors are fa r or inconsistent,
/u1D464fact(/u1D465)decreases toward zero, signaling that the memory provides u nreliable support. Thus
/u1D464fact(/u1D465)provides a continuous, geometry-aware assessment of the re liability of retrieval at the
query location /u1D465.
Mixture model. At the core of our framework lies a gated mixture that blends t he base LM with
the retriever. For each query /u1D465, predictions are produced by a convex combination of the LM a nd
the retriever distribution:
/u1D45D/u1D706(/u1D466|/u1D465)=(1−/u1D706(/u1D465))/u1D45E0(/u1D466|/u1D465) +/u1D706(/u1D465)ˆ/u1D45F(/u1D458)
/u1D466(/u1D465), /u1D466∈/u1D4B4, (2)
where the (measurable) gate /u1D706:R/u1D451→ [0,1]controls the relative reliance on retrieval. Small
values of /u1D706(/u1D465)favor the LM (ﬂuency and generalization), while values near one defer to retrieved
evidence (grounding and factuality). Intermediate values achieve an adaptive trade-oﬀ.
Population objective. The population-level loss balances predictive accuracy wi th trust-
dependent regularization:
ℒ(/u1D706)=E/bracketleftBig/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D44B) (−log/u1D45D/u1D706(/u1D466|/u1D44B))/bracketrightBig
+/u1D701E/bracketleftBig
/u1D706(/u1D44B) (1−/u1D464fact(/u1D44B))/bracketrightBig
, (3)

where/u1D443/u1D44C|/u1D44Bis the true conditional distribution and /u1D701/greaterorequalslant0. The ﬁrst term is the expected cross-
entropy, enforcing predictive ﬁt. The second penalizes ret rieval in regions with weak geometric
support (where /u1D464fact(/u1D465)is small). The hyperparameter /u1D701controls how strongly the gate penalizes
retrieval when the geometric support around the query is wea k.
When the query /u1D465is surrounded by close and coherent neighbors, /u1D464fact(/u1D465) ≈1, so the penalty
term/u1D706(/u1D465)(1−/u1D464fact(/u1D465))becomes negligible and the gate’s decision is governed prim arily by the
cross-entropy comparison between the LM and the retriever. In low-density or out-of-distribution
regions,/u1D464fact(/u1D465)becomes small, amplifying the penalty and discouraging rel iance on retrieval.
The mechanism therefore self-regulates: retrieval is driv en by predictive ﬁt, while geometric
support acts as a safeguard, allowing retrieval to compete f reely with the LM when support is
strong and otherwise pushing /u1D706(/u1D465)toward zero.
3. P/e.pc/r.pc-/q.pc/u.pc/e.pc/r.pc/y.pc /o.pc/p.pc/t.pc/i.pc/m.pc/i.pc/z.pc/a.pc/t.pc/i.pc/o.pc/n.pc /a.pc/n.pc/d.pc /h.pc/a.pc/r.pc/d.pc /g.pc/a.pc/t.pc/i.pc/n.pc/g.pc
The population loss ( 3) can be written as an expectation over the query space, ℒ(/u1D706)=
E[/u1D43D(/u1D706;/u1D44B)],where, for each ﬁxed /u1D465∈R/u1D451,
/u1D43D(/u1D706;/u1D465)=ℓ(/u1D706;/u1D465) +/u1D701/u1D706(/u1D465) (1−/u1D464fact(/u1D465))andℓ(/u1D706;/u1D465)=/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log/u1D45D/u1D706(/u1D466|/u1D465)).
Becauseℒ(/u1D706)is an expectation of the pointwise objective /u1D43D(/u1D706;/u1D465), and/u1D706enters/u1D43D(/u1D706;/u1D465)only
through its value at the query /u1D465, minimizing ℒ(/u1D706)reduces to minimizing /u1D43D(/u1D706;/u1D465)independently
for each/u1D465.
In the simplest and most interpretable implementation, the gate takes binary values /u1D706(/u1D465) ∈
{0,1}, corresponding to a sharp choice between the LM and the retri ever. The mixture model ( 2)
then selects one of its two components:
/u1D45D/u1D706(· |/u1D465)=/braceleftBigg
/u1D45E0(· |/u1D465), /u1D706(/u1D465)=0,
ˆ/u1D45F(/u1D458)(· |/u1D465), /u1D706(/u1D465)=1.
Deﬁne the corresponding local cross-entropies:
ℓ0(/u1D465)=/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log/u1D45E0(/u1D466|/u1D465))andℓ/u1D45F(/u1D465)=/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)).
Because under hard gating the decision /u1D706(/u1D465) ∈ {0,1}selects either the LM or the retriever, the
per-query objective
/u1D43D(/u1D706;/u1D465)=/braceleftBigg
ℓ0(/u1D465), /u1D706 (/u1D465)=0,
ℓ/u1D45F(/u1D465) +/u1D701(1−/u1D464fact(/u1D465)), /u1D706(/u1D465)=1,
reduces to a simple two-point comparison. The optimal gate a t/u1D465is therefore the choice that yields
the smaller of these two costs.
P/r.pc/o.pc/p.pc/o.pc/s.pc/i.pc/t.pc/i.pc/o.pc/n.pc 1 (O/p.pc/t.pc/i.pc/m.pc/a.pc/l.pc /h.pc/a.pc/r.pc/d.pc /g.pc/a.pc/t.pc/e.pc).For each query /u1D465∈R/u1D451, the Bayes-optimal hard gate is
/u1D706★(/u1D465)=/braceleftBigg
0,ifℓ0(/u1D465)/lessorequalslantℓ/u1D45F(/u1D465) +/u1D701(1−/u1D464fact(/u1D465)),
1,ifℓ/u1D45F(/u1D465) +/u1D701(1−/u1D464fact(/u1D465))< ℓ0(/u1D465).
The rule of Proposition 1states that retrieval is selected exactly when its improvem ent in pre-
dictive cross-entropy over the LM exceeds the geometric pen alty/u1D701(1−/u1D464fact(/u1D465)). The decision
therefore depends jointly on model ﬁt and neighborhood qual ity. When the retriever distribution

ˆ/u1D45F(/u1D458)(· |/u1D465)achieves a substantially lower cross-entropy than the LM and/u1D464fact(/u1D465)is large (i.e.,
the neighborhood of /u1D465is dense), the gate switches to retrieval. Conversely, if ne ighbors are far,
/u1D464fact(/u1D465)is small, the penalty dominates, and the gate keeps /u1D706★(/u1D465)=0. The parameter /u1D701acts as
a global regularizer: large values suppress retrieval and f avor the LM, while small values permit
more aggressive grounding in memory.
Soft gating. Although binary switching oﬀers direct interpretability, a continuous gate /u1D706(/u1D465) ∈
[0,1]may also be considered. Since −log/u1D45D/u1D706(/u1D466|/u1D465)is convex in /u1D706, the per-query objective /u1D43D(/u1D706;/u1D465)
is convex, and in fact strictly convex whenever /u1D45E0(· |/u1D465)and ˆ/u1D45F(/u1D458)(· |/u1D465)diﬀer on the support of
/u1D443/u1D44C|/u1D44B(· |/u1D465). The optimal soft gate satisﬁes the ﬁrst-order condition
/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)ˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D45E0(/u1D466|/u1D465)
/u1D45D/u1D706★(/u1D466|/u1D465)+/u1D701(1−/u1D464fact(/u1D465))=0,
which admits an eﬃcient one-dimensional numerical solutio n. In practice, however, the hard
decision rule of Proposition 1captures the essential structure of the gating mechanism an d
facilitates theoretical analysis of how retrieval, geomet ry, and factual reliability interact.
4. H/a.pc/l.pc/l.pc/u.pc/c.pc/i.pc/n.pc/a.pc/t.pc/i.pc/o.pc/n.pc /a.pc/n.pc/d.pc /d.pc/i.pc/s.pc/c.pc/o.pc/r.pc/d.pc/a.pc/n.pc/c.pc/e.pc /a.pc/n.pc/a.pc/l.pc/y.pc/s.pc/i.pc/s.pc
Hallucination arises when the LM produces conﬁdent predict ions that contradict evidence
present in retrieved neighbors. To quantify this phenomeno n, we combine the /u1D458-NN retrieval
distribution ˆ /u1D45F(/u1D458)
/u1D466(/u1D465)with the retrieval-trust weight /u1D464fact(/u1D465)deﬁned in ( 1). For each /u1D465∈R/u1D451, let
/u1D466/u1D45F(/u1D465) ∈arg max/u1D466∈/u1D4B4ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)
denote the retriever’s modal label (ties broken determinis tically). We deﬁne the local discordance
score as
ℋdisc(/u1D45E0;/u1D465)=/u1D464fact(/u1D465) (1−/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)),
and the associated population measure
ℋdisc(/u1D45E0)=E[ℋdisc(/u1D45E0;/u1D44B)].
This criterion is large when the retrieval neighborhood is g eometrically reliable ( /u1D464fact(/u1D465) ≈1) but
the LM assigns low probability to the label favored by retrie val. In this regime, the LM prediction
conﬂicts with locally supported evidence, which we interpr et as a risk of hallucination.
4.1. Change under optimal gating
Under the mixture model ( 2), the hallucination score becomes
ℋdisc(/u1D45D/u1D706;/u1D465)=/u1D464fact(/u1D465) (1−/u1D45D/u1D706(/u1D466/u1D45F(/u1D465) |/u1D465)).
Thus, the variation relative to the frozen LM is
Δℋ(/u1D465;/u1D706)=ℋdisc(/u1D45E0;/u1D465) −ℋdisc(/u1D45D/u1D706;/u1D465)
=/u1D706(/u1D465)/u1D464fact(/u1D465) (ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)), (4)
which is linear in /u1D706(/u1D465)and satisﬁes |Δℋ(/u1D465;/u1D706)|/lessorequalslant/u1D706(/u1D465)/u1D464fact(/u1D465).So, the sign of Δℋ(/u1D465;/u1D706)indicates
whether gating reduces ( /greaterorequalslant0) or increases (/lessorequalslant0) local discordance.
Recall that, under hard gating, the optimal decision rule fr om Proposition 1is
/u1D706★(/u1D465)=1{ℓ/u1D45F(/u1D465)+/u1D701(1−/u1D464fact(/u1D465))<ℓ0(/u1D465)}, (5)

whereℓ0(/u1D465)andℓ/u1D45F(/u1D465)are the LM and retriever local cross-entropy, respectively . Substituting ( 5)
into ( 4) yields the realized pointwise change
Δℋ(/u1D465;/u1D706★)=1{ℓ/u1D45F(/u1D465)+/u1D701(1−/u1D464fact(/u1D465))<ℓ0(/u1D465)}/u1D464fact(/u1D465) (ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)). (6)
Interpretation via three regimes. Equation ( 6) reveals three qualitatively distinct behaviors
governing how gating aﬀects hallucination.
(i) Gain region. On
/u1D49C={ℓ/u1D45F+/u1D701(1−/u1D464fact)< ℓ0,ˆ/u1D45F(/u1D458)
/u1D466/u1D45F/greaterorequalslant/u1D45E0(/u1D466/u1D45F)},
the retriever achieves a lower cross-entropy and assigns hi gher (or equal) mass to its own top
label than the LM. Retrieval therefore improves predictive ﬁt while reinforcing factual evidence,
implying Δℋ(/u1D465;/u1D706★)/greaterorequalslant0. The improvement is largest when /u1D464fact(/u1D465) ≈1 and is bounded above
byℋdisc(/u1D45E0;/u1D465).
(ii) Trade-oﬀ region. On
ℬ={ℓ/u1D45F+/u1D701(1−/u1D464fact)< ℓ0,ˆ/u1D45F(/u1D458)
/u1D466/u1D45F< /u1D45E0(/u1D466/u1D45F)},
the retriever improves the cross-entropy but places less ma ss on its modal label than the LM.
Here gating increases discordance ( Δℋ(/u1D465;/u1D706★)/lessorequalslant0), exhibiting a fundamental tension between
likelihood and factual alignment. The penalty /u1D701(1−/u1D464fact(/u1D465))mitigates these cases: when retrieval
is geometrically unreliable, the penalty rises and suppres ses harmful switches.
(iii) No-switch region. On
/u1D49E={ℓ/u1D45F+/u1D701(1−/u1D464fact)/greaterorequalslantℓ0},
the LM remains active ( /u1D706★(/u1D465)=0) andΔℋ(/u1D465;/u1D706★)=0. This region corresponds to sparse or
out-of-distribution queries where retrieval cannot overc ome its geometric penalty.
In summary, /u1D464fact(/u1D465)plays a dual role: it boosts the beneﬁt of switching in high-c onﬁdence
regions through its multiplicative factor and simultaneou sly reduces the risk of harmful switches
by amplifying the penalty where retrieval is unreliable.
4.2. Asymptotic analysis of discordance
The decomposition in the previous section showed that the po intwise change
Δℋ(/u1D465;/u1D706★)=/u1D706★(/u1D465)/u1D464fact(/u1D465) (ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465))
is entirely governed by the inner quantity
Δ(/u1D465)=ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465),
whose sign determines whether the optimal gate reduces or in creases the local hallucination
score. Its interpretation diﬀers sharply across the switch ing regions /u1D49Candℬ. On/u1D49C,Δ(/u1D465)/greaterorequalslant0
corresponds to a desirable gain: retrieval improves the loc al cross-entropy and assigns higher mass
to its own top label. On ℬ, however, we have Δ(/u1D465)<0 despite a cross-entropy improvement, and
it is unclear whether this reﬂects a genuine semantic disagr eement between Bayes and the LM,
or merely ﬁnite-sample variability of the /u1D458-NN estimator.
To clarify this, we analyze the asymptotic behavior of Δℋ(/u1D465;/u1D706★). Recall that we consider the
aligned setting, where the query and memory distributions c oincide,/u1D443/u1D44B=/u1D444/u1D448and/u1D443/u1D44C|/u1D44B=/u1D444/u1D449|/u1D448.
The ﬁnite-sample mode stability results established below imply that, when /u1D458→ ∞ and/u1D458//u1D45B→0,

the retriever distribution ˆ /u1D45F(/u1D458)(· |/u1D465)converges in probability to the Bayes conditional distribu tion
/u1D443/u1D44C|/u1D44B(· |/u1D465), and the induced modal label /u1D466/u1D45F(/u1D465)converges in probability to the Bayes label /u1D466★(/u1D465).
Because/u1D464fact(/u1D465) →1 on the support, the asymptotic behavior of Δℋ(/u1D465;/u1D706★)is then determined
solely by the structural diﬀerence
/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465).
As a consequence, any persistent negativity of Δℋ(/u1D465;/u1D706★)onℬmust be structural—stemming
from a true mismatch between the Bayes rule and the LM at /u1D465—rather than a by-product of /u1D458-NN
ﬂuctuations.
For notational convenience, let /u1D436:=|/u1D4B4|. For/u1D465∈R/u1D451and/u1D458≥1, we denote by
/u1D445/u1D458(/u1D465)=max
1/lessorequalslant/u1D457/lessorequalslant/u1D458/bardbl/u1D448(/u1D457)(/u1D465) −/u1D465/bardbl
the/u1D458-nearest-neighbor radius of the query /u1D465among the database points.
P/r.pc/o.pc/p.pc/o.pc/s.pc/i.pc/t.pc/i.pc/o.pc/n.pc 2 (F/i.pc/n.pc/i.pc/t.pc/e.pc-/s.pc/a.pc/m.pc/p.pc/l.pc/e.pc /m.pc/o.pc/d.pc/e.pc /s.pc/t.pc/a.pc/b.pc/i.pc/l.pc/i.pc/t.pc/y.pc ).Fix/u1D465∈supp(/u1D444/u1D448)and assume that the con-
ditional distribution /u1D443/u1D44C|/u1D44B(· | ·) is/u1D43F-Lipschitz in its second argument. Then, for all /u1D6FF∈ (0,1),
P/parenleftBig
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barex> /u1D6FF/parenrightBig
/lessorequalslant2/u1D436exp/parenleftBig
−2/u1D458/parenleftBig
/u1D6FF
2/parenrightBig2/parenrightBig
+P/parenleftBig
/u1D445/u1D458(/u1D465)>/u1D6FF
2/u1D43F/parenrightBig
.
In particular, if /u1D458→ ∞ and/u1D458//u1D45B→0as/u1D45B→ ∞ , then
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barexP−→0.
This proposition provides a uniform ﬁnite-sample bound on t he deviation ˆ /u1D45F(/u1D458)(· |/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)
at a ﬁxed query /u1D465, valid whenever /u1D443/u1D44C|/u1D44Bis locally Lipschitz and /u1D465lies in the support of the retrieval
distribution. In particular, the proposition shows that, a s soon as /u1D458grows while /u1D458//u1D45B→0, the
/u1D458-NN estimate concentrates around its Bayes target uniforml y over labels. This uniform control
is precisely what is needed to guarantee that, for large samp les, the empirical ordering of the
coordinates of ˆ /u1D45F(/u1D458)(· |/u1D465)matches the ordering of the Bayes vector /u1D443/u1D44C|/u1D44B(· |/u1D465). The next corollary
makes this consequence explicit by showing that the empiric al top label /u1D466/u1D45F(/u1D465)converges in
probability to the Bayes-optimal label /u1D466★(/u1D465)whenever the latter is unique.
C/o.pc/r.pc/o.pc/l.pc/l.pc/a.pc/r.pc/y.pc 1 (A/s.pc/y.pc/m.pc/p.pc/t.pc/o.pc/t.pc/i.pc/c.pc /m.pc/o.pc/d.pc/e.pc /s.pc/t.pc/a.pc/b.pc/i.pc/l.pc/i.pc/t.pc/y.pc ).Fix/u1D465∈supp(/u1D444/u1D448)and assume that the Bayes
label/u1D466★(/u1D465) ∈arg max /u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)is unique, so that
/u1D6FE(/u1D465):=max
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) − max
/u1D466≠/u1D466★(/u1D465)/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)>0.
Then, under the conditions of Proposition 2, if/u1D458→ ∞ and/u1D458//u1D45B→0as/u1D45B→ ∞ ,
P(/u1D466/u1D45F(/u1D465)≠/u1D466★(/u1D465)) −→ 0.
Asymptotic behavior of the local discordance Δ(/u1D465).We are in a position to analyze the large-
sample behavior of the key quantity Δ(/u1D465), which determines the sign and magnitude of the
hallucination variation Δℋ(/u1D465;/u1D706★)in (6).
Fix/u1D465∈supp(/u1D444/u1D448)and assume that the Bayes label /u1D466★(/u1D465)is unique, so that /u1D6FE(/u1D465)>0. By
Proposition 2, if/u1D458→ ∞ and/u1D458//u1D45B→0, then
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barexP−→0.

Therefore
ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466/u1D45F(/u1D465) |/u1D465)P−→0.
Recalling Δ(/u1D465)=ˆ/u1D45F(/u1D458)
/u1D466/u1D45F(/u1D465)(/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465), this implies
Δ(/u1D465) −/parenleftbig/u1D443/u1D44C|/u1D44B(/u1D466/u1D45F(/u1D465) |/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)/parenrightbigP−→0.
Moreover, Corollary 1yieldsP/parenleftbig/u1D466/u1D45F(/u1D465)=/u1D466★(/u1D465)/parenrightbig−→1. On the event {/u1D466/u1D45F(/u1D465)=/u1D466★(/u1D465)},
/u1D443/u1D44C|/u1D44B(/u1D466/u1D45F(/u1D465) |/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)=/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465).Hence,
/u1D443/u1D44C|/u1D44B(/u1D466/u1D45F(/u1D465) |/u1D465) −/u1D45E0(/u1D466/u1D45F(/u1D465) |/u1D465)P−→/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465),
and combining with the previous display gives
Δ(/u1D465)P−→/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465). (7)
In particular, the sign of Δ(/u1D465)coincides with the sign of /u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465)with
probability tending to one.
Asymptotic behavior of the local hallucination variation. We are now equipped to describe the
asymptotic behavior of the local hallucination variation Δℋ(/u1D465;/u1D706★)in (6). The result below shows
that, for ﬁxed /u1D465in the support, the sign and magnitude of Δℋ(/u1D465;/u1D706★)converge in probability to
a deterministic quantity governed solely by the Bayes predi ctor and the LM. We let
ℓBayes(/u1D465)=/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465))
and recall that the LM local cross-entropy is ℓ0(/u1D465)=/summationtext.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log/u1D45E0(/u1D466|/u1D465)).
T/h.pc/e.pc/o.pc/r.pc/e.pc/m.pc 1 (A/s.pc/y.pc/m.pc/p.pc/t.pc/o.pc/t.pc/i.pc/c.pc /b.pc/e.pc/h.pc/a.pc/v.pc/i.pc/o.pc/r.pc /o.pc/f.pc /t.pc/h.pc/e.pc /l.pc/o.pc/c.pc/a.pc/l.pc /h.pc/a.pc/l.pc/l.pc/u.pc/c.pc/i.pc/n.pc/a.pc/t.pc/i.pc/o.pc/n.pc /v.pc/a.pc/r.pc/i.pc/a.pc/t.pc/i.pc/o.pc/n.pc ).Under the
aligned setting, let /u1D465∈supp(/u1D444/u1D448). Suppose that the Bayes label /u1D466★(/u1D465) ∈arg max /u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)
is unique. Assume moreover that /u1D443/u1D44C|/u1D44B(· | ·) is/u1D43F-Lipschitz in its second argument and that
ℓBayes(/u1D465)≠ℓ0(/u1D465). Then, if /u1D458→ ∞ and/u1D458//u1D45B→0as/u1D45B→ ∞ ,
Δℋ(/u1D465;/u1D706★)P−→/u1D706∞(/u1D465) (/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465)),
where/u1D706∞(/u1D465):=1{ℓBayes(/u1D465)<ℓ0(/u1D465)}.
Thus, under mode stability, the local hallucination variat ionΔℋ(/u1D465;/u1D706★)converges in probability
to a deterministic quantity determined solely by the struct ural relationship between the Bayes
rule and the LM at /u1D465. In the limit, the randomness of both the /u1D458-NN estimator and the factuality
weight/u1D464fact(/u1D465)disappears, and the gating decision becomes entirely gover ned by the comparison
between the Bayes cross-entropy ℓBayes(/u1D465)and the LM cross-entropy ℓ0(/u1D465).
More precisely, when the switching condition ℓBayes(/u1D465)< ℓ0(/u1D465)holds, the gate eventually
activates with probability tending to one, and
Δℋ(/u1D465;/u1D706★) −→/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465).
WhenℓBayes(/u1D465)/greaterorequalslantℓ0(/u1D465), the gate eventually remains oﬀ, so that the variation Δℋ(/u1D465;/u1D706★) →0
even if the diﬀerence /u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D45E0(/u1D466★(/u1D465) |/u1D465)is negative.
In either case, any nonvanishing improvement or deteriorat ion of the hallucination score in the
large-sample regime reﬂects a genuine structural relation ship between the Bayes predictor and
the LM at /u1D465, rather than ﬁnite-sample instability of the /u1D458-NN estimator.

R/e.pc/f.pc/e.pc/r.pc/e.pc/n.pc/c.pc/e.pc/s.pc
G´erard Biau and Luc Devroye. Lectures on the Nearest Neighbor Method . Springer, Cham, 2015.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoﬀmann, Trevor Cai, et al. Improving language models by retrieving
from trillions of tokens. In Kamalika Chaudhuri, Stefanie J egelka, Le Song, Csaba Szepesv ´ari, Gang Niu, and
Sivan Sabato, editors, Proceedings of the 39th International Conference on Machin e Learning , volume 162 of
Proceedings of Machine Learning Research , pages 2206–2240. PMLR, 2022.
William Fedus, Barret Zoph, and Noam Shazeer. Switch transf ormers: Scaling to trillion parameter models with
simple and eﬃcient sparsity. Journal of Machine Learning Research , 23(120):1–39, 2022.
Edouard Grave, Moustapha M Cisse, and Armand Joulin. Unboun ded cache model for online language modeling
with open vocabulary. In Isabelle Guyon, Ulrike von Luxburg , Samy Bengio, Hanna Wallach, Rob Fergus,
S. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems , volume 30,
pages 6044–6054. Curran Associates, Inc., 2017.
L´aszl´o Gy ¨orﬁ, Michael Kohler, Adam Krzy˙ zak, and Harro Walk. A Distribution-Free Theory of Nonparametric
Regression . Springer, New York, 2006.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hossei ni, et al. Atlas: Few-shot learning with retrieval augmente d
language models. Journal of Machine Learning Research , 24(251):1–43, 2023.
Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Ge oﬀrey I. Hinton. Adaptive mixtures of local experts. In
Neural Computation , volume 3, pages 79–87, 1991.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, et al. Surve y of hallucination in natural language generation. ACM
Computing Surveys , 55:248,1–38, 2023.
Adam Tauman Kalai and Santosh S. Vempala. Calibrated langua ge models must hallucinate. In Proceedings of the
56th Annual ACM Symposium on Theory of Computing , STOC 2024, pages 160–171, New York, 2024. Association
for Computing Machinery.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemo yer, and Mike Lewis. Generalization through memo-
rization: Nearest neighbor language models. In International Conference on Learning Representations , 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petro ni, et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranz ato, Raia Hadsell, Maria-Florina Balcan, and Haibin
Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 9459–9474. Curran Associates,
Inc., 2020.
Potsawee Manakul, Adian Liusie, and Mark Gales. SelfCheckG PT: Zero-resource black-box hallucination detection
for generative large language models. In Houda Bouamor, Jua n Pino, and Kalika Bali, editors, Proceedings of
the 2023 Conference on Empirical Methods in Natural Languag e Processing , page 9004–9017. Association for
Computational Linguistics, 2023.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, et al. FAc tScore: Fine-grained atomic evaluation of factual
precision in long form text generation. In Houda Bouamor, Ju an Pino, and Kalika Bali, editors, Proceedings of
the 2023 Conference on Empirical Methods in Natural Languag e Processing , pages 12076–12100. Association for
Computational Linguistics, 2023.
Noam Shazeer, Azalia Mirhoseini, Piotr Maziarz, Andy Davis , Quoc Le, Geoﬀrey Hinton, and Jeﬀ Dean. Outrageously
large neural networks: The sparsely-gated mixture-of-exp erts layer. In International Conference on Learning
Representations , 2017.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, et a l. REPLUG: Retrieval-augmented black-box language
models. In Kevin Duh, Helena Gomez, and Steven Bethard, edit ors,Proceedings of the 2024 Conference of
the North American Chapter of the Association for Computati onal Linguistics: Human Language Technologies
(Volume 1: Long Papers) , pages 8371–8384. Association for Computational Linguist ics, 2024.
Lakpa Tamang, Mohamed Reda Bouadjenek, Richard Dazeley, an d Sunil Aryal. Handling out-of-distribution data: A
survey. arXiv:2507.21160 , 2025.
Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seo ng Oh. Calibrating large language models using
their generations only. In Lun-Wei Ku, Andre Martins, and Vi vek Srikumar, editors, Proceedings of the 62nd
Annual Meeting of the Association for Computational Lingui stics (Volume 1: Long Papers) , pages 15440–15459.
Association for Computational Linguistics, 2024.
Frank F. Xu, Uri Alon, and Graham Neubig. Why do nearest neigh bor language models work? In Andreas Krause,
Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan S abato, and Jonathan Scarlett, editors, Proceedings
of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning
Research , pages 38325–38341. JMLR, 2023.
Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Loy Chen Chan ge. Domain generalization: A survey. IEEE
Transactions on Pattern Analysis and Machine Intelligence , 45:4396–4415, 2023.

A.P/r.pc/o.pc/o.pc/f.pc/s.pc
A.1. Proof of Proposition 2
Conditionally on the neighbor locations /u1D448(1)(/u1D465),...,/u1D448 (/u1D458)(/u1D465), the variables
/u1D44D/u1D457,/u1D466:=1{/u1D449(/u1D457)(/u1D465)=/u1D466}, /u1D466∈/u1D4B4,
are independent Bernoulli with means /u1D443/u1D44C|/u1D44B(/u1D466|/u1D448(/u1D457)(/u1D465)), and ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)=/u1D458−1/summationtext.1/u1D458
/u1D457=1/u1D44D/u1D457,/u1D466. Therefore, by
Hoeﬀding’s inequality and a union bound over all classes, fo r any/u1D6FF∈ (0,1),
P/parenleftBig
max
/u1D466∈/u1D4B4/barex/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −1
/u1D458/u1D458/summationdisplay.1
/u1D457=1/u1D443/u1D44C|/u1D44B(/u1D466|/u1D448(/u1D457)(/u1D465))/barex/barex/barex>/u1D6FF
2/barex/barex/barex/u1D448(1)(/u1D465),...,/u1D448 (/u1D458)(/u1D465)/parenrightBig
/lessorequalslant2/u1D436exp/parenleftBig
−2/u1D458/parenleftBig
/u1D6FF
2/parenrightBig2/parenrightBig
.(8)
Dropping the conditioning yields the same bound unconditio nally.
Next deﬁne the local modulus of continuity
/u1D714/u1D465(/u1D45F):=sup
/bardbl/u1D462−/u1D465/bardbl/lessorequalslant/u1D45Fmax
/u1D466∈/u1D4B4|/u1D443/u1D44C|/u1D44B(/u1D466|/u1D462) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)|.
Clearly,
max
/u1D466∈/u1D4B4/barex/barex/barex1
/u1D458/u1D458/summationdisplay.1
/u1D457=1/u1D443/u1D44C|/u1D44B(/u1D466|/u1D448(/u1D457)(/u1D465)) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barex/barex/lessorequalslant/u1D714/u1D465(/u1D445/u1D458(/u1D465)).
If/u1D443/u1D44C|/u1D44Bis/u1D43F-Lipschitz, then /u1D714/u1D465(/u1D45F)/lessorequalslant/u1D43F/u1D45F, and combining this with ( 8) yields
P/parenleftBig
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barex> /u1D6FF/parenrightBig
/lessorequalslant2/u1D436exp/parenleftBig
−2/u1D458/parenleftBig
/u1D6FF
2/parenrightBig2/parenrightBig
+P/parenleftBig
/u1D445/u1D458(/u1D465)>/u1D6FF
2/u1D43F/parenrightBig
. (9)
Finally, since /u1D465∈supp(/u1D444/u1D448), the local mass /u1D45D/u1D465,/u1D700:=/u1D444/u1D448(/u1D435(/u1D465,/u1D700))is strictly positive for every /u1D700 >0.
Because
{/u1D445/u1D458(/u1D465)> /u1D700}={Bin(/u1D45B,/u1D45D/u1D465,/u1D700)< /u1D458},
Chernoﬀ’s bound [e.g., Biau and Devroye ,2015, Chapter 20] gives, whenever /u1D458/lessorequalslant1
2/u1D45B/u1D45D/u1D465,/u1D700,
P/parenleftbig/u1D445/u1D458(/u1D465)> /u1D700/parenrightbig/lessorequalslantexp/parenleftBig
−/u1D45B/u1D45D/u1D465,/u1D700
8/parenrightBig
.
Because /u1D458//u1D45B→0 implies /u1D458/lessorequalslant1
2/u1D45B/u1D45D/u1D465,/u1D700for large /u1D45B, the right-hand side tends to zero. Combining this
with ( 9) gives the desired convergence.
A.2. Proof of Corollary 1
Fix/u1D465∈supp(/u1D444/u1D448)and assume that the Bayes label /u1D466★(/u1D465)is unique, so that /u1D6FE(/u1D465)>0. Set/u1D700:=/u1D6FE(/u1D465)
3. By
Proposition 2, applied with this choice of /u1D700, we obtain
P/parenleftBig
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barex> /u1D700/parenrightBig
−→0 as/u1D45B→ ∞,
whenever /u1D458→ ∞ and/u1D458//u1D45B→0. Now, deﬁne the event
/u1D434/u1D45B:=/braceleftBig
max
/u1D466∈/u1D4B4/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/barex/barex/lessorequalslant/u1D700/bracerightBig
,
so thatP(/u1D434/u1D450
/u1D45B) →0 as/u1D45B→ ∞ . On/u1D434/u1D45B, for any/u1D4661,/u1D4662∈/u1D4B4,
ˆ/u1D45F(/u1D458)
/u1D4661(/u1D465) −ˆ/u1D45F(/u1D458)
/u1D4662(/u1D465)/greaterorequalslant/u1D443/u1D44C|/u1D44B(/u1D4661|/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D4662|/u1D465) −2/u1D700.

In particular, taking /u1D4661=/u1D466★(/u1D465)and any/u1D4662≠/u1D466★(/u1D465)gives
/u1D443/u1D44C|/u1D44B(/u1D466★(/u1D465) |/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D4662|/u1D465)/greaterorequalslant/u1D6FE(/u1D465)=3/u1D700,
hence
ˆ/u1D45F(/u1D458)
/u1D466★(/u1D465)(/u1D465) −ˆ/u1D45F(/u1D458)
/u1D4662(/u1D465)/greaterorequalslant3/u1D700−2/u1D700=/u1D700 > 0.
Thus, on /u1D434/u1D45B,
ˆ/u1D45F(/u1D458)
/u1D466★(/u1D465)(/u1D465)>ˆ/u1D45F(/u1D458)
/u1D466(/u1D465) ∀/u1D466≠/u1D466★(/u1D465),
and therefore the empirical and Bayes top labels coincide:
/u1D466/u1D45F(/u1D465)=arg max
/u1D466∈/u1D4B4ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)=arg max
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)=/u1D466★(/u1D465).
Consequently,
P(/u1D466/u1D45F(/u1D465)≠/u1D466★(/u1D465))/lessorequalslantP(/u1D434/u1D450
/u1D45B) −→ 0 as/u1D45B→ ∞,
which establishes the claim.
A.3. Proof of Theorem 1
Since/u1D465∈supp(/u1D444/u1D448)and/u1D458//u1D45B→0, Proposition 3yields
/u1D464fact(/u1D465)P−→1. (10)
Next, recall that
/u1D706★(/u1D465)=1{ℓ/u1D45F(/u1D465)+/u1D701(1−/u1D464fact(/u1D465))<ℓ0(/u1D465)},
where
ℓ/u1D45F(/u1D465)=/summationdisplay.1
/u1D466∈/u1D4B4/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465) (−log ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)).
By Proposition 2and the convention 0 log 0 =0, continuity of log on labels with /u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)>0 implies
ℓ/u1D45F(/u1D465)P−→ℓBayes(/u1D465).
Using ( 10), we obtain
ℓ/u1D45F(/u1D465) +/u1D701(1−/u1D464fact(/u1D465))P−→ℓBayes(/u1D465).
SinceℓBayes(/u1D465)≠ℓ0(/u1D465)by assumption, the indicator
1{ℓ/u1D45F(/u1D465)+/u1D701(1−/u1D464fact(/u1D465))<ℓ0(/u1D465)}
converges in probability to
/u1D706∞(/u1D465)=1{ℓBayes(/u1D465)<ℓ0(/u1D465)}.
Combining this limit with ( 7) proves the theorem.
B. Q/u.pc/e.pc/r.pc/y.pc-/m.pc/e.pc/m.pc/o.pc/r.pc/y.pc /d.pc/i.pc/s.pc/t.pc/r.pc/i.pc/b.pc/u.pc/t.pc/i.pc/o.pc/n.pc /m.pc/i.pc/s.pc/m.pc/a.pc/t.pc/c.pc/h.pc
Throughout the main text, we worked in the aligned setting, i n which the distribution of query inputs
coincides with that of the memory inputs ( /u1D443/u1D44B=/u1D444/u1D448), and the conditional label mechanisms also agree
(/u1D443/u1D44C|/u1D44B=/u1D444/u1D449|/u1D448). In realistic retrieval-augmented systems, however, thi s alignment rarely holds. The retrieval
memoryℳ/u1D45B={(/u1D448/u1D456,/u1D449/u1D456)}/u1D45B
/u1D456=1is typically built from a large and possibly heterogeneous c orpus, whereas
incoming queries (/u1D44B,/u1D44C)may follow a distinct or temporally evolving distribution. This discrepancy gives

rise to two fundamental sources of mismatch: a geometric shi ft, where the distribution of query embeddings
diﬀers from that of stored items, and a semantic drift, where the mapping between inputs and labels in
memory no longer reﬂects that of the current environment.
Such deviations are well known in the broader machine-learn ing literature. The former corresponds
to covariate shift, while the latter is related to semantic s hift, both extensively studied in the context of
domain adaptation and domain generalization; see, for exam ple,Zhou et al. [2023],Tamang et al. [2025].
In RAG, these mismatches manifest concretely as the retriev al of outdated, irrelevant, or oﬀ-distribution
items—conditions strongly associated with factual halluc ination.
To analyze these eﬀects within a uniﬁed framework,we introd uce a hybrid geometric-semantic mismatch
model , in which the distribution of queries may diﬀer from that of t he memory both through geometric
deformation of the input space and through semantic misalig nment between memory labels and the true
query labels. This setting allows us to interpret the retrie val-trust weight /u1D464fact(/u1D465)as an implicit correction
mechanism that naturally downweights unreliable retrieva ls.
B.1. A hybrid geometric-semantic mismatch model
We characterize query-memory mismatch by distinguishing t he distribution of query inputs /u1D443/u1D44Bfrom
that of memory inputs /u1D444/u1D448, and the query labeling mechanism /u1D443/u1D44C|/u1D44Bfrom the memory labeling mechanism
/u1D444/u1D449|/u1D448. This perspective allows us to model both geometric distort ion in the embedding space and semantic
mismatch in the conditional labels.
Geometric shift. Queries are assumed to be generated from memory inputs throu gh a perturbation model:
/u1D44B=/u1D447(/u1D448)=/u1D448+/u1D709(/u1D448),
where/u1D448∼/u1D444/u1D448and/u1D709:R/u1D451→R/u1D451is a deformation function. The distribution /u1D443/u1D44B=/u1D447#/u1D444/u1D448thus represents
the law of queries obtained by displacing the memory inputs. When computing the /u1D458nearest neighbors of
a query/u1D465, the retrieved points /u1D448(1)(/u1D465),...,/u1D448 (/u1D458)(/u1D465)are the elements of {/u1D448/u1D456}/u1D45B
/u1D456=1that lie closest to /u1D465in the
embedding space. If /bardbl/u1D709(/u1D462)/bardblis small, the neighborhoods of /u1D465and/u1D462largely overlap; as /bardbl/u1D709(/u1D462)/bardblincreases,
retrieved neighbors become less representative of /u1D465, capturing the geometric component of the mismatch.
Label drift. Even when geometric distortion is negligible, the conditio nal relationship between features
and labels in memory may diﬀer from that of current queries. W e model this semantic deviation as a local
corruption process:
/u1D444/u1D449|/u1D448(/u1D466|/u1D462)=(1−/u1D70C(/u1D462))/u1D443/u1D44C|/u1D44B(/u1D466|/u1D462) +/u1D70C(/u1D462)/u1D460(/u1D466|/u1D462), 0/lessorequalslant/u1D70C(/u1D462) ≤1,
where/u1D70C(/u1D462)is a local corruption rate and /u1D460(· |/u1D462)an arbitrary spurious distribution. Thus the retrieval
distribution constructed from ℳ/u1D45Bapproximates a corrupted version of /u1D443/u1D44C|/u1D44B: the memory is reliable when
/u1D70C(/u1D462)is small and increasingly misleading as /u1D70C(/u1D462)grows.
Retrieval trust and interpretation. The two mechanisms combine into the coupled model
/u1D448∼/u1D444/u1D448, /u1D44B=/u1D447(/u1D448), /u1D449|/u1D448∼/u1D444/u1D449|/u1D448(· |/u1D448), /u1D44C|/u1D44B∼/u1D443/u1D44C|/u1D44B(· |/u1D44B). (11)
Both types of shift inﬂuence the reliability of retrieval, b ut in diﬀerent ways. The retrieval-trust weight
/u1D464fact(/u1D465), deﬁned in ( 1), measures the geometric compatibility between the query /u1D465and its retrieved
neighbors. Large geometric deformations /bardbl/u1D709(/u1D462)/bardblinﬂate the distances /bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardbland therefore directly
reduce/u1D464fact(/u1D465). By contrast, strong semantic corruption /u1D70C(/u1D462)does not aﬀect /u1D464fact(/u1D465)itself, but makes the
retrieved labels unreliable as proxies for the true query la bels, even when geometric proximity is high.
Thus/u1D464fact(/u1D465)should be interpreted as a geometry-aware indicator of retr ieval quality, while the retrieval
distribution captures the semantic reliability of the retr ieved labels under joint geometric and semantic
shift.
In the next subsection, we formalize these observations by c haracterizing the asymptotic behavior of
both the retrieval-trust weight /u1D464fact(/u1D465)and the/u1D458-NN retriever ˆ /u1D45F(/u1D458)(· |/u1D465)under mild regularity assumptions
on/u1D447and/u1D70C. Proposition 3shows that /u1D464fact(/u1D465)converges to a deterministic function of the distance betwe en
the query and the memory support, capturing geometric compa tibility, while Proposition 4establishes that

the retriever converges to the local memory label distribut ion at the nearest points of the support. Together,
these results give a precise statistical interpretation of the penalty term in ( 3): retrieval is discouraged
precisely in regions where geometric proximity is weak or wh ere the limiting retrieval distribution reﬂects
boundary or corrupted semantics.
B.2. Geometry of the trust weight under distribution shift
Let{(/u1D448/u1D456,/u1D449/u1D456)}/u1D45B
/u1D456=1be the memory pairs with /u1D448/u1D456∈R/u1D451, drawn i.i.d. from /u1D444/u1D448/u1D449. Denote by /u1D444/u1D448the marginal
law of the /u1D448/u1D456’s, and by /u1D446=supp(/u1D444/u1D448)its (closed) support. For any query /u1D465∈R/u1D451, we let
/u1D451(/u1D465,/u1D446)=inf
/u1D462∈/u1D446/bardbl/u1D465−/u1D462/bardbl.
(Since/u1D446is closed, this inﬁmum is attained.)
P/r.pc/o.pc/p.pc/o.pc/s.pc/i.pc/t.pc/i.pc/o.pc/n.pc 3 (A/s.pc/y.pc/m.pc/p.pc/t.pc/o.pc/t.pc/i.pc/c.pc /b.pc/e.pc/h.pc/a.pc/v.pc/i.pc/o.pc/r.pc /o.pc/f.pc /t.pc/h.pc/e.pc /t.pc/r.pc/u.pc/s.pc/t.pc /w.pc/e.pc/i.pc/g.pc/h.pc/t.pc ).Fix/u1D465∈R/u1D451. If/u1D458//u1D45B→0as/u1D45B→ ∞ ,
then
/u1D464fact(/u1D465)=1
/u1D458/u1D458/summationdisplay.1
/u1D457=1exp/parenleftbig− /bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardbl2/parenrightbiga.s.−→exp/parenleftbig−/u1D451(/u1D465,/u1D446)2/parenrightbig.
In particular, if /u1D465∈/u1D446, then/u1D464fact(/u1D465)a.s.−→1.
Under the geometric component of the hybrid model, /u1D44B=/u1D447(/u1D448)=/u1D448+/u1D709(/u1D448)with/u1D448∼/u1D444/u1D448, a query
/u1D465=/u1D447(/u1D462)satisﬁes/u1D451(/u1D465,/u1D446)/lessorequalslant/bardbl/u1D465−/u1D462/bardbl=/bardbl/u1D709(/u1D462)/bardbl. Therefore, Proposition 3yields the almost-sure limit
/u1D464fact(/u1D465)a.s.− −−− →
/u1D45B→∞exp/parenleftbig−/u1D451(/u1D465,/u1D446)2/parenrightbig/greaterorequalslantexp/parenleftbig− /bardbl/u1D709(/u1D462)/bardbl2/parenrightbig,
with equality whenever /bardbl/u1D465−/u1D462/bardbl=/u1D451(/u1D465,/u1D446).
P/r.pc/o.pc/p.pc/o.pc/s.pc/i.pc/t.pc/i.pc/o.pc/n.pc 4 (A/s.pc/y.pc/m.pc/p.pc/t.pc/o.pc/t.pc/i.pc/c.pc/s.pc /o.pc/f.pc /t.pc/h.pc/e.pc /u1D458-NN /r.pc/e.pc/t.pc/r.pc/i.pc/e.pc/v.pc/a.pc/l.pc /d.pc/i.pc/s.pc/t.pc/r.pc/i.pc/b.pc/u.pc/t.pc/i.pc/o.pc/n.pc ).Fix/u1D465∈R/u1D451. Assume that /u1D458→
∞and/u1D458//u1D45B→0as/u1D45B→ ∞ , and deﬁne
ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)=1
/u1D458/u1D458/summationdisplay.1
/u1D457=11{/u1D449(/u1D457)(/u1D465)=/u1D466}, /u1D466∈/u1D4B4.
Let/u1D446=supp(/u1D444/u1D448)and let
/u1D441/u1D446(/u1D465)={/u1D462∈/u1D446:/bardbl/u1D465−/u1D462/bardbl=/u1D451(/u1D465,/u1D446)}
be the (possibly non-singleton) set of nearest points in /u1D446to/u1D465.
(i)In-support point. If/u1D465∈/u1D446and/u1D444/u1D449|/u1D448(/u1D466| ·)is continuous at /u1D465, then
ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)a.s.−→/u1D444/u1D449|/u1D448(/u1D466|/u1D465).
(ii)Unique nearest point oﬀ support. If/u1D465∉/u1D446, the nearest set is a singleton /u1D441/u1D446(/u1D465)={/u1D462/u1D465}, and/u1D444/u1D449|/u1D448(/u1D466| ·)
is continuous at /u1D462/u1D465, then
ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)a.s.−→/u1D444/u1D449|/u1D448(/u1D466|/u1D462/u1D465).
(iii)Multiple nearest points. If/u1D444/u1D449|/u1D448(/u1D466| ·)is continuous on /u1D441/u1D446(/u1D465), then almost surely,
min
/u1D462∈/u1D441/u1D446(/u1D465)/u1D444/u1D449|/u1D448(/u1D466|/u1D462)/lessorequalslantlim inf
/u1D45B→∞ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)/lessorequalslantlim sup
/u1D45B→∞ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)/lessorequalslantmax
/u1D462∈/u1D441/u1D446(/u1D465)/u1D444/u1D449|/u1D448(/u1D466|/u1D462).
In particular, if /u1D444/u1D449|/u1D448(/u1D466|/u1D462)is constant on /u1D441/u1D446(/u1D465), then ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)converges to that common value.
Interpretation under the hybrid shift model. Proposition 4shows that the empirical retriever ˆ /u1D45F(/u1D458)(· |/u1D465)
consistently estimates the memory’s local label mechanism in the region of the database that is geomet-
rically closest to the query. When /u1D465∈/u1D446, the estimate converges to /u1D444/u1D449|/u1D448(· |/u1D465); when/u1D465∉/u1D446but admits

a unique projection /u1D462/u1D465∈/u1D441/u1D446(/u1D465), it converges to /u1D444/u1D449|/u1D448(· |/u1D462/u1D465). Proposition 3complements this semantic
characterization with a geometric one: /u1D464fact(/u1D465) ≈1 indicates that /u1D465lies in or close to /u1D446, whereas small
values of /u1D464fact(/u1D465)ﬂag out-of-support queries for which retrieval reﬂects bou ndary behavior rather than
genuine local structure. Thus, under geometric and semanti c mismatch, the pair (ˆ/u1D45F(/u1D458),/u1D464fact)jointly encodes
the reliability of retrieved evidence, a property that dire ctly supports and stabilizes the gating rule.
Propositions 3and4yield almost-sure pointwise limits for /u1D464fact(/u1D465)and—when the limit exists—for
ˆ/u1D45F(/u1D458)(· |/u1D465). In particular, in cases (i) and (ii) of Proposition 4, the/u1D458-NN retriever admits a deterministic
limit/u1D45F∞(· |/u1D465). Whenever /u1D45F∞(· |/u1D465)exists, continuity of the mixture implies that, for any meas urable
/u1D706:R/u1D451→ [0,1],
/u1D45D/u1D706(/u1D466|/u1D465)=(1−/u1D706(/u1D465))/u1D45E0(/u1D466|/u1D465) +/u1D706(/u1D465)ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)
a.s.− −−− →
/u1D45B→∞/u1D45D/u1D706,∞(/u1D466|/u1D465):=(1−/u1D706(/u1D465))/u1D45E0(/u1D466|/u1D465) +/u1D706(/u1D465)/u1D45F∞(/u1D466|/u1D465).
Thus, under the hybrid shift model ( 11), we obtain
/u1D45F∞(/u1D466|/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)
=(1−/u1D70C(/u1D462/u1D465))/parenleftbig/u1D443/u1D44C|/u1D44B(/u1D466|/u1D462/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/parenrightbig+/u1D70C(/u1D462/u1D465)/parenleftbig/u1D460(/u1D466|/u1D462/u1D465) −/u1D443/u1D44C|/u1D44B(/u1D466|/u1D465)/parenrightbig,
and, writing /bardbl · /bardbl 1=/summationtext.1
/u1D466∈/u1D4B4| · |,
/bardbl/u1D45F∞(· |/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)/bardbl1/lessorequalslant(1−/u1D70C(/u1D462/u1D465))/u1D6FFgeom(/u1D465) +/u1D70C(/u1D462/u1D465)/u1D6FFsem(/u1D465),
where
/u1D6FFgeom(/u1D465)=/bardbl/u1D443/u1D44C|/u1D44B(· |/u1D462/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)/bardbl1, /u1D6FF sem(/u1D465)=/bardbl/u1D460(· |/u1D462/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)/bardbl1.
If/u1D443/u1D44C|/u1D44B(· |/u1D465)is locally Lipschitz as a map from R/u1D451to(R/u1D436,/bardbl · /bardbl 1), then
/u1D6FFgeom(/u1D465)=/bardbl/u1D443/u1D44C|/u1D44B(· |/u1D462/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)/bardbl1/lessorequalslant/u1D43F/bardbl/u1D465−/u1D462/u1D465/bardbl/lessorequalslant/u1D43F/u1D451(/u1D465,/u1D446).
Hence the total retrieval bias is jointly driven by the geome tric displacement /u1D451(/u1D465,/u1D446)and the local corruption
level/u1D70C(/u1D462/u1D465). When/u1D465lies near the memory support and corruption is small, the lim iting retriever /u1D45F∞(· |
/u1D465)is close to the true conditional distribution, and /u1D464fact(/u1D465) ≈1 makes the penalty /u1D706(/u1D465)(1−/u1D464fact(/u1D465))
negligible, so retrieval is not discouraged. As /u1D465moves away from the support or corruption increases,
/bardbl/u1D45F∞(· |/u1D465) −/u1D443/u1D44C|/u1D44B(· |/u1D465)/bardbl1grows while /u1D464fact(/u1D465) ≈/u1D452−/u1D451(/u1D465,/u1D446)2shrinks, naturally steering the gate toward the
base model.
B.3. Proof of Proposition 3
Let/u1D437(/u1D457)
/u1D45B(/u1D465)=/bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardblbe the/u1D457-th nearest-neighbor distance among {/u1D448/u1D456}/u1D45B
/u1D456=1. By Lemma 2.2 of
Biau and Devroye [2015], if/u1D458//u1D45B→0 then/u1D437(/u1D458)
/u1D45B(/u1D465)a.s.−→/u1D451(/u1D465,/u1D446)as/u1D45B→ ∞ . Since/u1D437(1)
/u1D45B(/u1D465)/lessorequalslant···/lessorequalslant/u1D437(/u1D458)
/u1D45B(/u1D465)
and/u1D437(1)
/u1D45B(/u1D465)/greaterorequalslant/u1D451(/u1D465,/u1D446), we obtain
0/lessorequalslantmax
1/lessorequalslant/u1D457/lessorequalslant/u1D458/barex/barex/u1D437(/u1D457)
/u1D45B(/u1D465) −/u1D451(/u1D465,/u1D446)/barex/barex/lessorequalslant/u1D437(/u1D458)
/u1D45B(/u1D465) −/u1D451(/u1D465,/u1D446)a.s.−→0,
so/u1D437(/u1D457)
/u1D45B(/u1D465)a.s.−→/u1D451(/u1D465,/u1D446)uniformly over 1/lessorequalslant/u1D457/lessorequalslant/u1D458. With/u1D711(/u1D461)=exp(−/u1D4612)continuous, uniform convergence
gives
max
1/lessorequalslant/u1D457/lessorequalslant/u1D458/barex/barex/u1D711(/u1D437(/u1D457)
/u1D45B(/u1D465)) −/u1D711(/u1D451(/u1D465,/u1D446))/barex/barexa.s.−→0.
This shows the desired claim.
B.4. Proof of Proposition 4
Let/u1D437(/u1D457)
/u1D45B(/u1D465)=/bardbl/u1D465−/u1D448(/u1D457)(/u1D465)/bardbl. As in the proof of Proposition 3, when/u1D458//u1D45B→0 as/u1D45B→ ∞ ,
0/lessorequalslantmax
1/lessorequalslant/u1D457/lessorequalslant/u1D458|/u1D437(/u1D457)
/u1D45B(/u1D465) −/u1D451(/u1D465,/u1D446)|/lessorequalslant/u1D437(/u1D458)
/u1D45B(/u1D465) −/u1D451(/u1D465,/u1D446)a.s.−→0.

Thus the neighbor locations satisfy almost surely: if /u1D465∈/u1D446, then/u1D448(/u1D457)(/u1D465) →/u1D465; if/u1D465∉/u1D446and/u1D441/u1D446(/u1D465)={/u1D462/u1D465},
then/u1D448(/u1D457)(/u1D465) →/u1D462/u1D465; in general, every cluster point of the sequence {/u1D448(/u1D457)(/u1D465)}/u1D458
/u1D457=1lies in/u1D441/u1D446(/u1D465).
Conditionally on the neighbor locations, the variables /u1D44D/u1D457,/u1D466:=1{/u1D449(/u1D457)(/u1D465)=/u1D466}are independent Bernoulli
with means /u1D444/u1D449|/u1D448(/u1D466|/u1D448(/u1D457)(/u1D465)), and ˆ/u1D45F(/u1D458)
/u1D466(/u1D465)=/u1D458−1/summationtext.1/u1D458
/u1D457=1/u1D44D/u1D457,/u1D466. Thus, Hoeﬀding’s inequality yields
P/parenleftBig/barex/barex/barexˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −1
/u1D458/u1D458/summationdisplay.1
/u1D457=1/u1D444/u1D449|/u1D448(/u1D466|/u1D448(/u1D457)(/u1D465))/barex/barex/barex> /u1D700/barex/barex/barex/u1D448(1)(/u1D465),...,/u1D448 (/u1D458)(/u1D465)/parenrightBig
/lessorequalslant2/u1D452−2/u1D458/u1D7002,
hence, as /u1D458→ ∞ ,
ˆ/u1D45F(/u1D458)
/u1D466(/u1D465) −1
/u1D458/u1D458/summationdisplay.1
/u1D457=1/u1D444/u1D449|/u1D448(/u1D466|/u1D448(/u1D457)(/u1D465))a.s.−→0.
For (i), continuity at /u1D465∈/u1D446implies
max
1/lessorequalslant/u1D457/lessorequalslant/u1D458|/u1D444/u1D449|/u1D448(/u1D466|/u1D448(/u1D457)(/u1D465)) −/u1D444/u1D449|/u1D448(/u1D466|/u1D465)|a.s.−→0,
so the Ces `aro mean converges to /u1D444/u1D449|/u1D448(/u1D466|/u1D465). The same argument gives (ii).
For (iii), continuity on /u1D441/u1D446(/u1D465)implies the Ces `aro averages of {/u1D444/u1D449|/u1D448(/u1D466|/u1D448(/u1D457)(/u1D465))}/u1D458
/u1D457=1must lie within
the convex hull of the function values on /u1D441/u1D446(/u1D465), giving the stated bounds.