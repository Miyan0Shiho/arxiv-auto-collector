# Predicate Importance Estimation and Decoupled Rationale-Score Distillation for Entity Alignment

**Authors**: Keunha Kim, Yoonjin Jang, Hyeon-gu Lee, Sihyung Kim, Youngjoong Ko

**Published**: 2026-06-22 08:11:35

**PDF URL**: [https://arxiv.org/pdf/2606.22992v1](https://arxiv.org/pdf/2606.22992v1)

## Abstract
Knowledge graphs (KGs) are increasingly used as structured context for Large Language Models (LLMs), but industrial KG-RAG systems often need to integrate public and domain-specific KGs constructed from heterogeneous databases. This integration relies on Entity Alignment (EA), where lexical matching alone is insufficient under predicate-name variation and incomplete local neighborhoods. We address EA for KG integration by constructing a pairwise EA dataset and proposing two complementary modules: Predicate Importance Estimation (PIE) and Decoupled Rationale-Score Distillation (DRSD). PIE is a compact embedding-based approach that removes the subject information from each 1-hop triple, encodes the resulting subjectless triples, and aggregates them with learnable predicate-importance weights to build predicate-aware entity embeddings. DRSD trains a distilled small language model (SLM) with pseudo-answers produced by a teacher LLM through distinct prompts. By converting binary EA labels into text-based supervision and decoupling confidence-score estimation from label-consistent rationales, DRSD enables the SLM to learn task-specific reasoning while retaining a less label-biased confidence signal. Experiments show that PIE and DRSD improve EA classification. Moreover, because DRSD decouples confidence-score estimation from the decision, a discrepancy between the two flags an uncertain prediction for human review, thereby enabling a practical discrepancy between automatic acceptance and human-in-the-loop verification.

## Full Text


<!-- PDF content starts -->

Predicate Importance Estimation and Decoupled Rationale-Score
Distillation for Entity Alignment
Keunha Kim1, Yoonjin Jang1, Hyeon-gu Lee2, Sihyung Kim2, Youngjoong Ko1*
1SungKyunKwan University,2NA VER,
{keunhakim98, yoonjinjang98}@gmail.com, yjko@skku.edu
{hyeongu.lee, sihyung.kim}@navercorp.com
Abstract
Knowledge graphs (KGs) are increasingly used
as structured context for Large Language Mod-
els (LLMs), but industrial KG-RAG systems of-
ten need to integrate public and domain-specific
KGs constructed from heterogeneous databases.
This integration relies on Entity Alignment
(EA), where lexical matching alone is insuf-
ficient under predicate-name variation and in-
complete local neighborhoods. We address EA
for KG integration by constructing a pairwise
EA dataset and proposing two complemen-
tary modules: Predicate Importance Estimation
(PIE) and Decoupled Rationale-Score Distil-
lation (DRSD). PIE is a compact embedding-
based approach that removes the subject infor-
mation from each 1-hop triple, encodes the re-
sulting subjectless triples, and aggregates them
with learnable predicate-importance weights
to build predicate-aware entity embeddings.
DRSD trains a distilled small language model
(SLM) with pseudo-answers produced by a
teacher LLM through distinct prompts. By con-
verting binary EA labels into text-based super-
vision and decoupling confidence-score estima-
tion from label-consistent rationales, DRSD en-
ables the SLM to learn task-specific reasoning
while retaining a less label-biased confidence
signal. Experiments show that PIE and DRSD
improve EA classification. Moreover, because
DRSD decouples confidence-score estimation
from the decision, a discrepancy between the
two flags an uncertain prediction for human re-
view, thereby enabling a practical discrepancy
between automatic acceptance and human-in-
the-loop verification.
1 Introduction
Knowledge graphs (KGs) are increasingly used as
structured knowledge sources for Large Language
Models (LLMs) (Pan et al., 2024; Sun et al., 2024).
Industrial systems simultaneously reason over pub-
lic KGs such as DBpedia and Wikidata, as well
*Corresponding author
Figure 1: 1-hop graph example for an entity-alignment
pair
as domain-specific KGs that are constructed from
multiple internal relational databases via standard
mappings (Das et al., 2012; Sequeda and Miranker,
2017). Therefore, these heterogeneous sources
need to be integrated into a unified KG. To build
such a unified KG, duplicate entities must be iden-
tified and merged across sources, which requires an
explicit entity-alignment step.Entity Alignment
(EA) identifies equivalent entities across distinct
KGs (Sun et al., 2020b; Zhang et al., 2022). EA is
intrinsically hard for two reasons that matter espe-
cially when systems bridge different graphs. First,
two graphs typically use different predicate-naming
conventions (e.g., employer_of vs.worksFor ), so
purely structural or string-based matching is mis-
led by surface variance. Second, knowledge graphs
collected for different purposes often contain dif-
ferent types and amounts of information. There-
fore, robust EA must go beyond semantic simi-
larity through joint modeling of the surrounding
subgraph and therelative importanceof each pred-
icate as evidence for alignment.
To handle such complex semantic judgments,
Large Language Models (LLMs) offer attractive
generalization capabilities, and these capabilities
have led recent work to exploit them for EA (Jiang
et al., 2024; Zhang et al., 2024; Chen et al., 2024).
1arXiv:2606.22992v1  [cs.CL]  22 Jun 2026

However, these LLM-based aligners rely heavily
on general pretraining knowledge. As a result, they
struggle to internalize the very domain-specific re-
lations and user-defined alignment constraints that
define these heterogeneous graphs because such
domain-specific patterns were under-represented
during pretraining.
In this work, we address entity alignment for KG
integration by constructing a pairwise EA dataset
and introducing two complementary alignment
modules.Our datasetreflects the pre-alignment
setting: each instance contains two ambiguous can-
didate entities and their 1-hop subgraph evidence,
so the model must decide equivalence using local
graph context.
Our first module, Predicate Importance Esti-
mation(PIE), is a compact embedding-based ap-
proach that removes the subject information from
each 1-hop triple, independently encodes the sub-
jectless triples, and aggregates the representations
with learnable predicate-importance weights. As
a result, PIE constructs a compact subgraph-level
entity embedding, estimates which predicates are
most useful for EA, and achieves strong entity
alignment performance.
Our second module, Decoupled Rationale-
Score Distillation (DRSD), trains a distilled
small language model (SLM) using decoupled
rationale-score pseudo-answers produced by an
LLM from distinct prompts. Since binary la-
bels lack the reasoning and confidence signals
needed for text-generating models, DRSD uses
task-specific prompts to elicit structured teacher
responses and convert EA labels into text-based
supervision. When eliciting responses from the
teacher LLM through prompts, the gold-label
prompt produces a label-consistent decision and
rationale, whereas the score prompt asks the LLM
to estimate similarity without exposing the gold la-
bel or requiring a generated decision. The final SFT
target combines the gold-label decision and justi-
fication with the score prompt confidence score,
allowing the student model to learn task-specific
reasoning while retaining a less label-biased confi-
dence signal.
Compared with the prompt-only LLM baseline,
both proposed modules improve EA performance
when trained on the constructed dataset. PIE im-
proves Acc. and F1 by 6.32 and 4.63 percentage
points, and DRSD improves them by 7.66 and 6.76
percentage points. Moreover, because DRSD de-
couples the confidence score from the decision, dis-agreements between the two can flag predictions to
route to human review (13.3% of pairs), which can
further increase F1 from 0.886 to 0.909.
2 Related Work
2.1 Entity Alignment and LLM-based
Approaches
Entity Alignment (EA) has been studied through
embedding-based and neighborhood-aware meth-
ods, including translation-style KG embeddings,
cross-graph extensions, and GNN-based aligners
(Sun et al., 2020b; Zhang et al., 2022; Chen et al.,
2017; Sun et al., 2018; Wang et al., 2018; Wu
et al., 2019; Sun et al., 2020a). These methods
show that structural, attribute, and predicate-aware
features are important for distinguishing equivalent
entities across heterogeneous KGs (Trisedya et al.,
2019; Xu et al., 2019; Zhang et al., 2022). More
recent LLM-based approaches, such as ChatEA,
AutoAlign, and LLM4EA, exploit the semantic rea-
soning ability of LLMs for EA (Jiang et al., 2024;
Zhang et al., 2024; Chen et al., 2024). However,
they can be brittle on domain-specific relations and
costly to run for each candidate pair (Ling et al.,
2023). Our work complements these lines of re-
search by learning predicate-aware entity repre-
sentations and distilling LLM alignment reasoning
into a smaller model with a score-based confidence
signal.
2.2 Knowledge Distillation from LLMs to
Small Models
Knowledge distillation (Hinton et al., 2015) has
recently been extended from logit-matching to ra-
tionale transfer.Distilling Step-by-Step(Hsieh
et al., 2023) trains a small model jointly on LLM-
generated labels and chain-of-thought rationales
(Wei et al., 2022) to match teacher performance
with far fewer parameters. Subsequent work (Mag-
ister et al., 2023; Ho et al., 2023) demonstrates
that small models can be taught to reason via CoT
distillation, and SCOTT (Wang et al., 2023) intro-
duces a counterfactual objective to reduce shortcut
imitation.
2.3 Predicate-Aware KG Embeddings
Beyond the translation-based and GNN-based
methods above, several lines of work explicitly
model relation or attribute importance. KBGAT
(Nathani et al., 2019) introduces attention-based
neighborhood weighting for embeddings; relation-
2

aware attention is similarly central to MRAEA
(Mao et al., 2020) and RDGCN (Wu et al., 2019);
and attribute-centric methods such as JAPE (Sun
et al., 2017) and AttrE (Trisedya et al., 2019) treat
predicates and literal attributes as first-class signals.
PIE instead uses subjectless, predicate-aware rep-
resentations with importance-weighted pooling to
produce a compact entity signature.
3 Preliminaries
LetC={G 1, . . . , G M}denote a collection of
knowledge graphs (KGs). Each KG is defined as
Gm= (E m,Pm,Tm),(1)
where Emis the entity set, Pmis the predicate
set, and Tmis the triple set of the m-th KG. We
assume that entities are unique within each KG,
while equivalent entities may appear across differ-
ent KGs.
Our task is to compare entities from two different
KGs, GmandGn(m̸=n ), and decide whether
they denote the same entity. E.g., e≡e′ore̸≡
e′,e∈G m, e′∈G n. Each triple is denoted as
(s, p, o) , where s,p, andodenote subject, predicate,
and object, respectively. For an entity e∈ E m, its
outgoing 1-hop neighborhoodN m(e)is
Nm(e) ={(s, p, o)∈ T m|s=e}.(2)
For a cross-KG entity pair (e, e′)∈ E m× En, we
compute its similarity scores e,e′as
se,e′=ϕ 
Nm(e),N n(e′)
,(3)
where ϕ(·,·) is an entity-pair scoring function. The
final alignment decisionby e,e′is then
bye,e′=(
1,ifs e,e′≥τ,
0,otherwise,(4)
where τis a decision threshold, bye,e′= 1indicates
e≡e′, andby e,e′= 0indicatese̸≡e′.
Thus the task is to retrieve similar cross-
collection candidates and decide match/non-match
for integration.
4 Methodology
Two complementary components are used for entity
alignment. The first one is Predicate Importance
Estimation (PIE), a lightweight statistical module
that learns which predicates are most informative
for entity alignment. The second one is DecoupledRationale-Score Distillation (DRSD), a reasoning-
aware distillation framework that transfers decision
patterns from a large teacher model to a smaller
student model.
4.1 Predicate Importance Estimation
For the 1-hop neighborhood Nm(e)of an entity e,
Predicate Importance Estimation (PIE) constructs
a representative embedding vefrom local rela-
tional evidence. It first encodes subjectless triples,
then pools triple representations by predicate, and
finally aggregates predicate-level representations
with predicate-importance weights. This process
yields both an entity embedding and predicate-
importance weights, and the weights indicate which
predicates are most informative for alignment. PIE
consists of three stages: (1) subjectless triple encod-
ing, (2) predicate-wise pooling, and (3) predicate-
importance pooling.
Subjectless triple encoding.For each outgoing
triple (e, p, o)∈ N m(e), we remove the subject
and encode only the predicate–object text. This re-
duces reliance on the surface name of eand makes
entity comparison depend more on relational evi-
dence:
h(e)
p,o= Enc 
text(p)⊕text(o)
,(5)
where ⊕denotes text concatenation. Let P(e) =
{p| ∃o: (e, p, o)∈ N m(e)} be the set of predi-
cates around e, and let Op(e) ={o|(e, p, o)∈
Nm(e)}be the object set for predicatep.
Predicate-wise pooling.Since an entity may
have multiple triples with the same predicate, we
apply mean pooling over their triple embeddings
to obtain one predicate-level representation:
r(e)
p=1
|Op(e)|X
o∈O p(e)h(e)
p,o.(6)
Predicate-importance pooling.We then aggre-
gate the predicate-level representations into a single
entity embedding:
αp=exp(w p)P
p′∈P(e) exp(w p′),
ve=X
p∈P(e)αpr(e)
p.(7)
Here, wpis a learnable scalar parameter for pred-
icate p, and its softmax-normalized value αprep-
resents the relative importance of that predicate
withinP(e).
3

Figure 2: Overview of the proposed framework. (Left) Entity embedding construction ( ve) within PIE; embeddings
for each entity pair are subsequently concatenated for MLP-based classification. (Right) Illustrative example of the
decoupled rationale-score distillation process.
Entity-pair training.For a candidate pair (e, e′),
we concatenate the two entity embeddings and feed
them into an MLP classifier:
ˆye,e′=σ 
MLP([v e;ve′])
,(8)
where σ(·) is the sigmoid function and ˆye,e′∈[0,1]
denotes the predicted match probability. We train
the classifier with binary cross-entropy over entity
pairs with labels, where ye,e′= 1 indicates that
eande′are equivalent, and ye,e′= 0 otherwise.
During training, we freeze the pretrained encoder
and update only the predicate-importance weights
and the MLP classifier layers. This keeps the train-
ing overhead low and quickly adapts the predicate-
importance weights to the alignment task.
4.2 Decoupled Rationale-Score Distillation
Conventional LLM distillation trains students to im-
itate teacher outputs, which makes the teacher an
implicit performance ceiling when its responses are
incorrect. DRSD instead separates pseudo-answer
generation into a supervised signal, where the
teacher sees the gold label, and an unsupervised sig-
nal, where it does not. Combining the two preserves
label-guided reasoning while yielding a confidence
score that is less tied to the supervised answer.
Decoupled Rationale-Score Label Construction
Our goal is to construct pseudo-answers that pre-
serve the teacher’s reasoning ability while prevent-
ing the confidence score from being biased toward
the gold label in the prompt, regardless of the ob-
jective similarity between the two entities. For the
i-th labeled entity pair, let Xidenote the serialized
evidence for the two entities and yiits gold label.We use three prompt schemas:
qgold
i= Promptgold(Xi, yi),
qscore
i= Promptscore(Xi),
qsft
i= Promptsft(Xi).(9)
The gold-label prompt generates a supervised de-
cision and rationale, whereas the score prompt es-
timates confidence from the evidence without re-
ceiving the gold label or producing a decision. The
sft prompt is a simplified student training prompt
for efficient fine-tuning. The teacher outputs are
ogold
i=T(qgold
i) = 
jgold
i, dgold
i, cgold
i
,
oscore
i=T(qscore
i) = 
jscore
i, cscore
i
,(10)
where d,c, andjdenote the decision, score, and jus-
tification fields, respectively. Appendix C provides
the prompt templates and representative JSON-
style teacher responses.
We then compose the final SFT target with the
decision and justification from the gold-label out-
put and the score from the score prompt output:
ℓsft
i= 
dgold
i, cscore
i, jgold
i
.(11)
These three fields correspond to the final alignment
decision, confidence score, and reasoning justifica-
tion, respectively. This decoupled rationale-score
construction uses label information where it is most
reliable—the decision and justification—and keeps
confidence estimation independent of the gold la-
bel in the prompt. As a result, the score can be
used as a more objective thresholding signal for
precision–recall trade-offs and human-in-the-loop
review.
4

4.2.1 Predicate-Importance Weights
The predicate-importance weights learned by PIE
are also used to compress LLM inputs. For a budget
k, we keep only the triples whose predicates fall in
the top-kset,
TopK(α, k) = arg max
S⊆P(e),|S|=kX
p∈Sαp,(12)
˜Nm(e) ={(e, p, o)∈ N m(e)|p∈TopK(α, k)},
(13)
where αpis the predicate-importance weight from
Eq. (7). We use k=10 and further retain at most 10
objects per predicate (Appendix B.3), which pre-
serves informative evidence while reducing prompt
length and inference cost.
4.2.2 SFT
We construct an SFT dataset for a smaller language
model as prompt–pseudo-answer pairs:
DSFT={(qsft
i, ℓsft
i)}N
i=1.(14)
The student model is fine-tuned to generate the de-
coupled rationale-score target response from the
sft prompt. In this way, the small language model
learns the gold-label decision and justification, and
it inherits the prompt-derived score as its confi-
dence signal. At deployment time, the generated
score can be thresholded to trade precision against
recall according to application constraints. To an-
alyze the effect of score decoupling, we compare
DRSD againstLabel-Guided Distillation (LGD),
which is an ablation that uses a single gold-label
prompt to jointly produce the decision, rationale,
and score without decoupling.
5 Experiments
5.1 Datasets & Models
DatasetsWe evaluate our method on a real-
world, industrial-scale labeled entity-pair dataset
constructed from our in-house heterogeneous
knowledge graphs. The dataset contains a total of
85,858 training entity pairs and 5,000 evaluation
pairs. Each pair is assigned a binary label:SAME
for equivalent entities andDIFFERENTfor non-
equivalent entities. A detailed description of the
underlying knowledge graphs and dataset statistics,
including 1-hop triple distributions, is provided in
Appendix A.Method Acc. Prec. Rec. F1
LLM 0.7976 0.7433 0.9092 0.8179
SLM 0.7418 0.7737 0.6836 0.7258
PIE(ours) 0.8608 0.8438 0.8856 0.8642
DRSD(ours) 0.8742 0.8126 0.9728 0.8855
+ Human Review0.9020 0.8463 0.9824 0.9093
Table 1: Entity alignment performance on the evaluation
split. LLM and SLM are prompt-only; DRSD is fine-
tuned from the SLM.+ Human Reviewcorrects the
13.3% of pairs flagged as score–decision conflicts.
ModelsFor PIE, we use BAAI/bge-m31to
generate embeddings and employ a two-layer
MLP as the classifier. For DRSD, we use
openai/gpt-oss-120b2as the teacher LLM and
train a hcx-srch1-8B (SLM) for distillation. De-
tailed architecture and training hyperparameters are
provided in Appendix B.
5.2 Results
Table 1 reports entity alignment performance on
the evaluation split. For DRSD, the model outputs
both a decision and a similarity score. we compute
the reported metrics from the decision output by
treatingSAMEpredictions as the positive label.
The results highlight two main findings. First,
PIE achieves strong performance despite its com-
pact architecture and small number of trainable pa-
rameters. This suggests that subjectless triples can
be effectively encoded in the semantic embedding
space and aggregated with predicate-aware weights
to construct discriminative entity representations
for classification.
Second, the comparison between LLM and
DRSD shows that pseudo-answer distillation is
effective. The distilled SLM outperforms the
prompted LLM and implicitly learns task-specific
alignment rules embedded in the data. We note
that DRSD inherits a prior distribution biased to-
wardSAMEdue to the imbalanced training split
(76.8%SAME) and thus prioritizes recall, whereas
PIE exhibits more balanced performance. Finally,
routing DRSD’s uncertain predictions to human
review raises F1 from 0.886 to 0.909 (Table 1); we
analyze why this is possible in Section 5.3.
5.3 Analysis
Predicate Budget kTable 2 reports prompt-
only teacher LLM classification performance as
1https://huggingface.co/BAAI/bge-m3
2https://huggingface.co/openai/gpt-oss-120b
5

kAcc. Prec. Rec. F1
5 0.7956 0.73400.92720.8194
10 0.7976 0.7433 0.9092 0.8179
15 0.8044 0.7532 0.9056 0.8224
200.8094 0.75740.91040.8269
Table 2: LLM classification results across top kpredi-
cate budgets. Best scores are shown in bold.
Figure 3: Predicate-importance weights learned by PIE
for predicates in the specific domain (Company).
the predicate budget kin PIE’s importance-based
compression varies (Section 4.2.1). Performance
changes only marginally as kgrows from 10 to 20,
while prompt length increases with k. This indi-
cates that the top-ranked predicates already carry
most of the alignment-relevant evidence, which
serves as the basis for selecting k=10 to minimize
the input volume while maintaining accuracy.
Predicate-importance WeightsFigure 3 vi-
sualizes the predicate-importance weights that
PIE learns in the company domain. The highly
weighted predicates correspond to attributes and
relations describing organizational identity or busi-
ness context, indicating that PIE learns inter-
pretable predicate preferences consistent with hu-
man judgment, beyond merely improving classifi-
cation.
LGD vs DRSDFigure 4 compares where LGD
and DRSD place their errors across score-bins
(false negatives at top, false positives at bottom). In
LGD, the score is tied to the decision.DIFFERENT
predictions receive low scores andSAMEpredic-
tions receive high scores, so both false negatives
and false positives for this model fall within the ex-
pected side of the threshold. As a result, LGD pro-
duces almost no score–decision discrepancy, and
no signal indicates which predictions are unreli-
able. DRSD instead decouples the score from the
decision, which can assign high scores to 24 pairs
predicted asDIFFERENTand low scores to 115
pairs predicted asSAME.These discrepancies concentrate on the model’s
errors, and form a concrete criterion for human-
in-the-loop review. Only these conflicting cases
(13.3% of the evaluation set) need to be routed
to a human annotator. The same review applied
to LGD yields almost no change, since it pro-
duces no discrepancies to act on. In an industrial
KG-integration environment where false-positive
merges result in significant costs and review capac-
ity is limited, this allows an operator to automati-
cally accept confident predictions and focus their
review efforts solely on the small set of genuinely
uncertain ones that a label-coupled score cannot
support.
00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.90102030CountFalse Negative (FN)
LGD
DRSD
00.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.90100200300CountFalse Positive (FP)
Figure 4: Score-bin distribution of false negatives
(top) and false positives (bottom). DRSD shows score–
decision discrepancies (high-score FN, low-score FP)
that can route to human review, whereas LGD produces
almost none.
6 Conclusion
We presented the entity-alignment framework for
heterogeneous KG integration that combines Predi-
cate Importance Estimation (PIE) and Decoupled
Rationale-Score Distillation (DRSD). PIE learns
predicate-aware entity representations from sub-
jectless 1-hop triples, while DRSD distills LLM
reasoning into a smaller model by decoupling label-
consistent rationales from confidence-score estima-
tion. Crucially, because DRSD’s score is decoupled
from its decision, discrepancies between decision
and score detect likely errors and provide a crite-
rion for human review that a label-coupled score
cannot offer. Correcting only these detected error
6

cases further improves F1. These results suggest
that efficient and operationally practical EA is fea-
sible by selecting task-relevant graph evidence with
a predicate-aware method and distilling LLM rea-
soning into a compact model. The decoupling prin-
ciple is not limited to EA and it can also be useful
in other human-in-the-loop distillation settings.
Limitations
Our study has several limitations. First, although
PIE is designed to learn predicate-importance
weights, these weights can become biased when
some classes or predicates dominate the train-
ing data. Ideally, predicate importance should
be estimated in a class-aware manner, but pred-
icate distributions also vary across collections.
Future work should explore multiple importance
weights, ensemble-based weighting, or other de-
biasing strategies that produce predicate weights
generalizing across both classes and collections.
Second, we do not provide a component-level
ablation of PIE, such as isolating the contribution
of subjectless encoding or comparing learnable im-
portance weights against uniform pooling. A finer-
grained analysis of these design choices is left to
future work.
Third, our human-review results assume correct
annotator judgment and therefore represent an up-
per bound on the achievable gain. In practice, the
realized improvement depends on annotator accu-
racy and the available review budget, and the 13.3%
of pairs flagged for review may be costly to inspect
at scale. Estimating the gain under realistic annota-
tor behavior is left to future work.
Ethical Considerations
Data.Our experiments use proprietary, in-house
knowledge graphs constructed from internal
databases. The data contains entity records such
as organizations, people, and creative works but
no private personal information beyond what is
publicly associated with these entities. Because the
data is proprietary, we do not release it; we instead
report dataset statistics and full implementation de-
tails (Appendix A) to support reproducibility of the
method.
Human review.The “+ Human Review” row in
Table 1 uses the manually verified evaluation la-
bels as the outcome of reviewing the flagged score–
decision conflicts. Because these labels are human-
verified, the result can be interpreted as a simula-tion of correct human review on the flagged subset.
We report it separately from fully automatic meth-
ods because deployment gains would still depend
on annotator accuracy, review capacity, and quality-
control procedures.
Intended use.The proposed method targets
knowledge-graph integration in industrial settings.
As with any automated alignment system, incorrect
merges could propagate errors into downstream
applications. The confidence-based routing we pro-
pose is designed precisely to mitigate this risk by
escalating uncertain cases to human review.
References
Muhao Chen, Yingtao Tian, Mohan Yang, and Carlo
Zaniolo. 2017. Multilingual knowledge graph embed-
dings for cross-lingual knowledge alignment. InPro-
ceedings of the Twenty-Sixth International Joint Con-
ference on Artificial Intelligence, IJCAI-17, pages
1511–1517.
Shengyuan Chen, Qinggang Zhang, Junnan Dong, Wen
Hua, Qing Li, and Xiao Huang. 2024. Entity align-
ment with noisy annotations from large language
models. InAdvances in Neural Information Process-
ing Systems (NeurIPS).
Souripriya Das, Seema Sundara, and Richard Cyganiak.
2012. R2RML: Rdb to rdf mapping language. W3C
Recommendation.
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.
Distilling the knowledge in a neural network.arXiv
preprint arXiv:1503.02531.
Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023.
Large language models are reasoning teachers. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 14852–14882, Toronto, Canada.
Association for Computational Linguistics.
Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh,
Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay
Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Dis-
tilling step-by-step! outperforming larger language
models with less training data and smaller model
sizes. InFindings of the Association for Compu-
tational Linguistics: ACL 2023, pages 8003–8017,
Toronto, Canada. Association for Computational Lin-
guistics.
Xuhui Jiang, Yinghan Shen, Zhichao Shi, Chengjin
Xu, Wei Li, Zixuan Li, Jian Guo, Huawei Shen, and
Yuanzhuo Wang. 2024. Unlocking the power of large
language models for entity alignment. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 7566–7583, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
7

Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan Deng,
Can Zheng, Junxiang Wang, Tanmoy Chowdhury,
Yun Li, Hejie Cui, Tianjiao Zhao, Amit Panalkar,
Wei Cheng, Haoyu Wang, Yanchi Liu, Zhengzhang
Chen, Haifeng Chen, Chris White, Quanquan Gu,
Carl Yang, and Liang Zhao. 2023. Domain special-
ization as the key to make large language models
disruptive: A comprehensive survey.arXiv preprint
arXiv:2305.18703.
Lucie Charlotte Magister, Jonathan Mallinson, Jakub
Adamek, Eric Malmi, and Aliaksei Severyn. 2023.
Teaching small language models to reason. InPro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 2: Short
Papers), pages 1773–1781, Toronto, Canada. Associ-
ation for Computational Linguistics.
Xin Mao, Wenting Wang, Huimin Xu, Man Lan, and
Yuanbin Wu. 2020. Mraea: An efficient and robust
entity alignment approach for cross-lingual knowl-
edge graph. InProceedings of the 13th International
Conference on Web Search and Data Mining, WSDM
’20, page 420–428, New York, NY , USA. Association
for Computing Machinery.
Deepak Nathani, Jatin Chauhan, Charu Sharma, and
Manohar Kaul. 2019. Learning attention-based em-
beddings for relation prediction in knowledge graphs.
InProceedings of the 57th Annual Meeting of the As-
sociation for Computational Linguistics, pages 4710–
4723, Florence, Italy. Association for Computational
Linguistics.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Ji-
apu Wang, and Xindong Wu. 2024. Unifying large
language models and knowledge graphs: A roadmap.
IEEE Transactions on Knowledge and Data Engi-
neering, 36(7):3580–3599.
Juan F. Sequeda and Daniel P. Miranker. 2017. A pay-as-
you-go methodology for ontology-based data access.
IEEE Internet Computing, 21(2):92–96.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. InInternational Confer-
ence on Learning Representations (ICLR).
Zequn Sun, Wei Hu, and Chengkai Li. 2017. Cross-
lingual entity alignment via joint attribute-preserving
embedding. InThe Semantic Web – ISWC 2017,
pages 628–644, Cham. Springer International Pub-
lishing.
Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu.
2018. Bootstrapping entity alignment with knowl-
edge graph embedding. InProceedings of the Twenty-
Seventh International Joint Conference on Artificial
Intelligence, IJCAI-18, pages 4396–4402. Interna-
tional Joint Conferences on Artificial Intelligence
Organization.Zequn Sun, Chengming Wang, Wei Hu, Muhao Chen,
Jian Dai, Wei Zhang, and Yuzhong Qu. 2020a.
Knowledge graph alignment network with gated
multi-hop neighborhood aggregation. InProceed-
ings of AAAI, pages 222–229.
Zequn Sun, Qingheng Zhang, Wei Hu, Chengming
Wang, Muhao Chen, Farahnaz Akrami, and Chengkai
Li. 2020b. A benchmarking study of embedding-
based entity alignment for knowledge graphs.Proc.
VLDB Endow., 13(12):2326–2340.
Bayu Distiawan Trisedya, Jianzhong Qi, and Rui Zhang.
2019. Entity alignment between knowledge graphs
using attribute embeddings. InProceedings of AAAI,
pages 297–304.
Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao,
Bing Yin, and Xiang Ren. 2023. SCOTT: Self-
consistent chain-of-thought distillation. InProceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 5546–5558, Toronto, Canada. Association for
Computational Linguistics.
Zhichun Wang, Qingsong Lv, Xiaohan Lan, and
Yu Zhang. 2018. Cross-lingual knowledge graph
alignment via graph convolutional networks. InPro-
ceedings of EMNLP, pages 349–357.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems,
volume 35, pages 24824–24837. Curran Associates,
Inc.
Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui
Yan, and Dongyan Zhao. 2019. Relation-aware en-
tity alignment for heterogeneous knowledge graphs.
InProceedings of the Twenty-Eighth International
Joint Conference on Artificial Intelligence, IJCAI-19,
pages 5278–5284. International Joint Conferences on
Artificial Intelligence Organization.
Kun Xu, Liwei Wang, Mo Yu, Yansong Feng, Yan Song,
Zhiguo Wang, and Dong Yu. 2019. Cross-lingual
knowledge graph alignment via graph matching neu-
ral network. InProceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics,
pages 3156–3161, Florence, Italy. Association for
Computational Linguistics.
Rui Zhang, Yixin Su, Bayu Distiawan Trisedya, Xi-
aoyan Zhao, Min Yang, Hong Cheng, and Jianzhong
Qi. 2024. Autoalign: Fully automatic and effective
knowledge graph alignment enabled by large lan-
guage models.IEEE Transactions on Knowledge
and Data Engineering, 36(6):2357–2371.
Rui Zhang, Bayu Distiawan Trisedya, Miao Li, Yong
Jiang, and Jianzhong Qi. 2022. A benchmark and
comprehensive survey on knowledge graph entity
alignment via representation learning.The VLDB
Journal, 31(5):1143–1168.
8

A Data Statistics
Knowledge GraphsOur entity pairs are drawn
from in-house heterogeneous knowledge graphs
spanning multiple domains and sources. The graphs
encompass entities from diverse domains such as
entertainment (films, music, and broadcasts), com-
panies, people, and cultural properties, connected
through 2,904 distinct relation types. Because the
graphs are heterogeneous, entities vary in their
types and in the relations connecting them, so the
set of 1-hop triples available for an entity differs
substantially in both quantity and schema across
the dataset. For each entity in a pair, we extract its
1-hop triples as the structural context provided to
the model.
StatisticsTable 3 reports the dataset scale, label
distribution, and 1-hop triple statistics for the train-
ing and evaluation splits. The labeled training split
is imbalanced towardSAMEpairs, which account
for 76.8% of training examples, whereas the eval-
uation split is intentionally balanced with a 50/50
label ratio. For the triple statistics, Triples/entity
(mean) indicates the average number of 1-hop
triples available per entity, while Triples/entity
(max) indicates the largest number of 1-hop triples
observed in each split.
Statistic Train Evaluation
Entity pairs 85,858 5,000
SAME65,984 (76.8%) 2,500 (50.0%)
DIFFERENT19,874 (23.2%) 2,500 (50.0%)
Unique entities 136,626 9,499
Triples/entity (mean) 26.17 25.0
Triples/entity (max) 4,349 1,811
Table 3: Dataset statistics for the labeled training and
evaluation splits. Triples/entity reports the mean and
maximum number of 1-hop triples per entity.
B Implementation Details
B.1 Decoupled Rationale-Score Distillation
We fine-tune an hcx-srch1-8B model as the stu-
dent SLM using the SFT objective described in
the main method, with DeepSpeed ZeRO Stage 2,
bfloat16 precision, and gradient checkpointing.
We train for 1 epoch with AdamW (learning rate
2×10−5, 100 warmup steps, no weight decay), an
effective batch size of 8, and a maximum sequence
length of 4096 tokens. We apply completion-only
loss masking, discard over-length samples, and use
greedy decoding with up to 256 new tokens at in-
ference time. During inference, the decision fieldis used for match classification, and the score field
is retained for threshold-based confidence analysis.
B.2 Predicate Importance Estimation (PIE)
All predicate–object texts are encoded offline with
theBAAI/bge-m3 encoder. For each candidate pair,
the resulting entity embeddings are concatenated
and passed to a two-layer MLP classifier. The pre-
trained encoder is frozen, and only the predicate-
importance weights and MLP classifier layers are
trained with binary cross-entropy using AdamW
(learning rate 1×10−3, 100 warmup steps, no
weight decay) for up to 20 epochs; validation accu-
racy selects the best checkpoint.
B.3 Pseudo-Answer Construction
Each entity’s 1-hop neighborhood is serialized
as JSON-style evidence, excluding collection-
type objects and retaining at most 10 objects
per predicate to control prompt length. For
each labeled candidate pair, the teacher LLM
(openai/gpt-oss-120b ) is queried with both the
gold-label prompt and the score prompt. The final
SFT target follows the main decoupled rationale-
score construction: the decision and justification
are taken from the gold-label output, while the
score is taken from the score prompt output.
9

C Prompt & Output Examples
(A) Gold-Label Prompt (GOLD_TEMPLATE)
You will receive two JSON descriptions, and each description is generated from an entity’s subgraph (attributes and
relations).
You will ALSO receive the manually verified gold (ground-truth) decision label.
Your task is to generate:
1) A well-calibrated similarity score in [0, 1]
2) A high-quality justification that is CONSISTENT with the given decision and the score that you generate
This output will be used for knowledge distillation.
Your task has TWO phases:
PHASE 1 (internal reasoning, not fully exposed):
- Use the entity instance type that the model infers to determine which features are CORE vs. AUXILIARY .
- Compare the two entities with type-appropriate core attributes and relations.
- Identify matches that provide support, explainable discrepancies, conflicts, and information that is missing.
- Choose a similarity score that:
- Is consistent with the evidence strength
- Respects the decision threshold:
- If decision = "SAME", score MUST be≥0.60
- If decision = "DIFFERENT", score MUST be<0.60
- Construct reasoning that supports BOTH the decision and the chosen score.
IMPORTANT:
- You MUST NOT contradict the gold decision that the prompt provides.
- The score that you generate MUST be consistent with the decision.
- Treat the gold decision as correct, even if the evidence is weak or ambiguous.
PHASE 2 (final output only):
- Output ONLY the simplified JSON that the template below defines.
- Do NOT include internal reasoning or mention the gold decision explicitly.
[Entity A Description]
{A_TEXT}
[Entity B Description]
{B_TEXT}
[Gold Label]
Decision: {GOLD_DECISION}
FINAL OUTPUT FORMAT (STRICT):
Return ONLY valid JSON in the following format, with no additional text.
{
"justification": [
"Brief reason 1 (most important type-specific evidence or explanation)",
"Brief reason 2",
"Brief reason 3"
],
"decision": "{GOLD_DECISION}",
"score": 0.0
}
Constraints:
- The decision MUST exactly match the gold decision that the prompt provides.
- The score MUST be a float in [0, 1] with at most two decimal places.
- Score constraints:
- If decision = "SAME"→score≥0.60
- If decision = "DIFFERENT"→score<0.60
- Justifications must be:
- Type-appropriate
- Consistent with both the decision and the score magnitude
- Do NOT add or remove fields.
Figure 5: Gold-label prompt used to generate a label-consistent decision and justification for distillation.
10

(B) Score Prompt (SCORE_TEMPLATE)
You will receive two JSON descriptions, and each description is generated from an entity’s subgraph (attributes and
relations).
Your task has TWO phases:
PHASE 1 (internal reasoning, not fully exposed):
- Compare the two entities with the most type-appropriate core attributes and relations.
- Identify high-confidence matches, matches that provide support, conflicts (high/medium/low), and information that is
missing or uncertain.
- conflicts should be weighted by their importance to the entity type.
- Apply the scoring principles below to compute a similarity score between 0 and 1.
PHASE 2 (final output only):
- Output ONLY the simplified JSON that the template below defines.
- Do NOT include the full internal analysis in the final answer.
- Unique identifiers>entity type>type-specific core attributes>key relations>auxiliary attributes.
- Type mismatch or unique ID conflicts should strongly reduce the score.
- Missing information increases uncertainty but is not a conflict.
[Entity A Description]
{A_TEXT}
[Entity B Description]
{B_TEXT}
FINAL OUTPUT FORMAT (STRICT):
Return ONLY valid JSON in the following format, with no additional text.
{
"justification": [
"Brief reason 1 (most important type-specific evidence or conflict)",
"Brief reason 2",
"Brief reason 3"
],
"score": 0.0
}
Constraints:
- The score must be a float in [0, 1] with at most two decimal places.
- Justification items must reflect features that are appropriate for the entity type that the model infers.
Figure 6: Score prompt used to estimate an evidence-based similarity score for an entity pair.
(C) SFT Prompt (SFT_TEMPLATE)
You will receive two JSON descriptions, and each description is generated from an entity’s subgraph (attributes and
relations).
Determine whether the two entities refer to the SAME entity or DIFFERENT entities.
[Entity A Description]
{A_TEXT}
[Entity B Description]
{B_TEXT}
Output:
- decision: SAME or DIFFERENT
- score: a similarity score in [0, 1]
- three brief justification points
Return only valid JSON in the specified format.
Figure 7: SFT prompt used as the student-model input at training and inference time.
11

(A) Gold-Label Output (JSON)
{
"justification": [
"Both entities share the title \"Memories\" but have different release years (2023 vs 2019).",
"The listed composers and performers differ completely (J.Fla and multiple Western composers vs
Maroon 5 and Korean-named composers).",
"Additional metadata such as record label, language, and related works are unrelated. This
indicates that they are distinct musical works."
],
"decision": "DIFFERENT",
"score": 0.22
}
Figure 8: Example gold-label prompt output containing a decision, score, and justification.
(B) Score Prompt Output (JSON)
{
"justification": [
"Both entities share the exact name \"Memories\" and overlapping core composers (Adam Levine and
Jonathan/Jon Bellion). This indicates the same musical work.",
"Entity A is a 2023 cover by J.Fla, whereas Entity B is the original 2019 recording by Maroon
5; the performer and release-date differences reflect different recordings, not a conflict on
composition identity.",
"Other attributes (record label, part_of album, instance_of) differ but are performance-specific
and do not outweigh the strong composer and title match."
],
"score": 0.78
}
Figure 9: Example score prompt output containing an evidence-based score and justification.
(C) SFT Target (JSON)
{
"justification": [
"Both entities share the title \"Memories\" but have different release years (2023 vs 2019).",
"The listed composers and performers differ completely (J.Fla and multiple Western composers vs
Maroon 5 and Korean-named composers).",
"Additional metadata such as record label, language, and related works are unrelated. This
indicates that they are distinct musical works."
],
"decision": "DIFFERENT",
"score": 0.78
}
Figure 10: Example SFT target combining the gold-label decision and justification with the score prompt score.
12