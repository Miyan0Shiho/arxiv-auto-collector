# Query-Aware Source-Risk Triage for Retrieval-Augmented Generation

**Authors**: Kainan Zhou, Gangzhen Qian, Chuhong Xu, Lu Yi

**Published**: 2026-09-15 03:01:27

**PDF URL**: [https://arxiv.org/pdf/2609.16564v1](https://arxiv.org/pdf/2609.16564v1)

## Abstract
Retrieval-augmented generation (RAG) pipelines may omit a source's material relationship to the query. We study a pre-generation triage layer that treats this relationship as query dependent. The method routes canonical query families for enhanced review and assigns retrieved pages to pass, contextualize, exclude, or review. It combines a four-dimension page score, rank-discounted family aggregation, intent-preserving query mutations, and a family-held-out router. A single-coded pilot of 200 real URLs supplies provisional calibration anchors; a 20,000-row scenario with synthetic domain identifiers supports controlled workload analysis. An oracle page gate defines a risk-coverage target for a future learned classifier. The evaluation shows why page-level frequency cannot substitute for family-level exposure and quantifies how calibration changes scenario activation. Annotation reliability remains unmeasured, and synthetic rankings omit real retrieval dynamics. The result is an auditable triage method and validation plan, not an estimate of deployed review workload, live-Web prevalence, or downstream answer-quality gains.

## Full Text


<!-- PDF content starts -->

Query-Aware Source-Risk Triage for
Retrieval-Augmented Generation
Kainan Zhou
Google LLC
Mountain View, USA
zhoumark@google.comGangzhen Qian
Google LLC
Mountain View, USA
irisqian@google.comChuhong Xu
Sony Corporate of America
San Jose, USA
chuhong.xu@sony.comLu Yi
Google LLC
Mountain View, USA
annabelyi@google.com
Abstract—Retrieval-augmented  generation  (RAG)  pipelines  
may omit a source's material relationship to the query. We study  
a pre-generation triage layer that treats this relationship as query  
dependent.  The  method  routes  canonical  query  families  for  
enhanced  review  and  assigns  retrieved  pages  to  pass,  
contextualize, exclude, or review. It combines a four-dimension  
page score, rank-discounted family aggregation, intent-preserving  
query mutations, and a family-held-out router. A single-coded  
pilot of 200 real URLs supplies provisional calibration anchors; a  
20,000-row scenario with synthetic domain identifiers supports  
controlled workload analysis. An oracle page gate defines a risk–
coverage target for a future learned classifier. The evaluation  
shows why page-level frequency cannot substitute for family-level  
exposure  and  quantifies  how  calibration  changes  scenario  
activation.  Annotation  reliability  remains  unmeasured,  and  
synthetic rankings omit real retrieval dynamics. The result is an  
auditable triage method and validation plan, not an estimate of  
deployed review workload, live-Web prevalence, or downstream  
answer-quality gains.
Keywords—retrieval-augmented generation, source provenance,  
query-aware triage, query clustering, third-party bias
I. INTRODUCTION
Retrieval-augmented generation (RAG)  combines  search,  
ranking, and synthesis. Verifiability audits distinguish citation  
completeness from whether each citation supports its associated  
statement [1]. Those tests are necessary, but they do not cover a  
separate source-side failure. A vendor comparison may report  
accurate specifications, an affiliate review may contain useful  
measurements, and an advocacy group may cite primary records.  
Trouble arises when synthesis removes this material relationship  
and presents evidence from an interested source as independent.
End-to-end evaluation must also separate retrieval, citation  
placement, and answer generation; ALCE evaluates all three and  
shows that fluent output can conceal upstream failure [2]. Source  
relationships arise in both ordinary and adversarial retrieval, but  
an interested source is not necessarily false or malicious. This  
distinction motivates examining provenance after retrieval and  
before generation.
Our two-stage framework first routes query families for  
enhanced provenance review and then defines query-page policy  
actions:  pass,  contextualize,  exclude,  or  review,  without  
permanently labeling publishers.
The evaluation uses  two complementary components. A  
manual audit of 200 real URLs supplies page-score anchors and  
records access failures. A frozen scenario of 400 constructed  query families and 20,000 query-page rows uses those anchors  
to test aggregation, threshold workload, controlled mutations,  
and  family-held-out  routing.  Because  the  manual  means  
calibrate  the  scenario,  agreement  between  the  two  is  by  
construction  rather  than  validation.  We  therefore  ask  three  
bounded questions: how should query-page relationship risk be  
represented,  how  does  manual  anchoring  affect  family  
activation, and what risk–coverage target should a later page  
classifier meet?
The  resulting  prototype  contributes  a  query-page  
relationship score with rank-aware family activation, a family-
held-out router test for controlled query reformulations, and an  
oracle risk–coverage target for a future learned page classifier.  
Fig. 1 summarizes the workflow. The present study does not  
include matched downstream answer generation.
Fig. 1.Query-aware  triage  workflow.  Manual  page  scores  calibrate  the  constructed scenario; family routing precedes query-page policy actions.
II. RELATED WORK AND SCOPE
A.RAG and Citation Evaluation
Citation quality requires more than reference presence: SIDE  
identifies claims needing citations and retrieves stronger sources  
for Wikipedia statements [3]. This motivates checking whether  
each retrieved source supports the attached factual claim.
Our scope is complementary to answer-level hallucination  
detection: it asks whether a supported passage comes from a  
materially interested source, while keeping factual support and  
provenance as separate review dimensions.
B.Source-Side Influence and Disclosure
Source-side  influence  can  arise  through  optimization,  
injected  documents,  affiliate  infrastructure,  incomplete  
disclosure,  coordination,  or  undisclosed  sponsorship.  These  

mechanisms  differ  in  intent,  so  our  signal  remains  query  
dependent and does not prove deception.
C.Unit and Claim Boundary
The unit of analysis is a query-page pair (q, u), not a person,  
publisher, or domain. Its score is a query-specific review-priority  
index, not the probability that u is false, deceptive, coordinated,  
or  malicious.  We  use  four  evaluation  strata:  product  
recommendation  (PROD),  software  or  service  comparison  
(SOFT), political or electoral information (POL), and your-
money-or-your-life topics (YMYL), including health, finance,  
or local services. These strata organize evaluation, not Web  
prevalence.
D.Threat Model and Operating Objective
The  method  addresses  retrieval  sets  where  relationship  
evidence may be visible on a page, in a redirect chain, or in  
public ownership records, yet absent from the excerpt delivered  
to a generator. A source may enter through ordinary ranking  
rather than adversarial insertion. The failure is contextual: the  
generator receives a useful passage but lacks information needed  
to interpret the source’s stake. We assume neither control of the  
retriever nor access to private ranking features. The objective is  
provenance review before synthesis while preserving sufficient  
evidence after policy actions. Deliberate poisoning is one case,  
but the same interface covers sponsored, affiliated, or advocacy  
material in ordinary retrieval.
III. METHODOLOGY
A.Manual Calibration Corpus
Each query-page pair receives four ordinal ratings in {0,  
0.25,  0.50,  0.75,  1}:  query-relevant  incentive  or  material  
relationship I, directional advocacy A, disclosure opacity O, and 
apparent independence P. We combine them as
Spage(q,u)=(IAOP)1/4.(1)
The geometric mean is deliberately conjunctive: any zero  
yields a zero composite. The score is an operational review-
priority  index;  a  low  value  does  not  certify  safety  or  
independence, nor is it a probability of falsehood, deception, or  
malicious intent.
The manual workbook contains 20 English audit queries,  
five  per  vertical,  with  ten  recorded  URLs  per  query.  Two  
reviewers divided the 200 pages equally and recorded source  
role, evidence notes, access status, and the four dimensions. No  
pages were independently double-rated, so reliability and inter-
rater agreement remain unmeasured. This is a single-coded pilot,  
not a human gold standard. Section VII specifies an overlap  
audit.  Reviewer-entered  titles  only  aid  navigation  and  are  
excluded from evidence.
Fourteen pages remain unresolved: six were unavailable,  
four led to the wrong page, and four were partial or paywalled.  
They remain missing rather than being imputed as safe. Most fall  
in one reviewer’s assignment, but the disjoint allocation prevents  
reviewer and access effects from being separated.B.Manual-Anchored Scenario Generator
The  scenario  uses  a  frozen  panel  of  400  constructed  
canonical queries, 100 per vertical. Each query has 50 ranked  
synthetic source observations, yielding 20,000 query-page rows.  
Synthetic  domain  identifiers  do  not  correspond  to  real  
publishers; no search engine, page crawl, model call, or human  
rating produced these rows. The generator includes topic, intent,  
domain, rank, and noise terms, with fixed rank component 0.22 
exp(−(r−1)/3.2) and fixed random seed 20260716.
We fit one additive intercept δv per vertical. For raw score  
Sraw and resolved manual mean μv, δv solves
E[clip(Sraw+δv,0,1)∣v)=μv.(2)
Only four intercepts are fitted; topic, query, rank, domain,  
and noise structure remain frozen. Raw and anchored scores  
are  both  retained.  Each  intercept  changes  only  its  vertical  
baseline,  so  subsequent  activation  changes  arise  from  the  
declared calibration step. The goal is to prevent arbitrary vertical  
offsets from dominating workload analysis, not to validate the  
generator against the records used for calibration.
C.Rank-Discounted Family Score
For a query with  nq ranked pages, ranks receive weights  
proportional to 1/log2(r+1), normalized within that query. Long-
context experiments show that evidence position affects model  
use,  which  motivates  rank-discounted  exposure  while  page  
policy remains local [4]. The family score is
Squery(q)=∑r=1nq
wrSpage(q,ur),∑r=1nq
wr=1.(3)
Here  nq = 10 for the manual audit and  nq = 50 for the 
constructed scenario. A family activates when Squery(q)  ≥ τ and 
at least two constituent pages also satisfy Spage(q,ur)  ≥ τ. The 
main analysis uses τ = 0.50 and repeats the same rule from 0.40  
to 0.60. The two-page guard prevents a single extreme page from  
determining  family  activation.  Page-level  policy  decisions  
remain local to each page within the originating query.
D.Controlled Mutations and Router
Each canonical query produces 12 textual forms: the original  
wording and 11 intent-preserving mutations  spanning seven  
categories:  comparison,  evidence-first,  skeptical,  disclosure-
aware, constraint-aware, alternative-seeking, and independent-
source. All forms from one canonical query remain in the same  
train, development, or test partition. The 60/20/20 family split  
therefore prevents near-duplicate formulations from crossing  
partitions.
We  compare  exact-string  canonical  matching,  a  
development-tuned risk lexicon, word term frequency–inverse  
document frequency (TF–IDF) logistic regression, and a word-
plus-character  TF–IDF  router.  The  prediction  target  is  the  
anchored scenario activation rule rather than a human query  
label.  Results  on  80  held-out  families  therefore  measure  
recovery  of  controlled  reformulations  under  that  rule,  not  
accuracy on natural or deployed queries.

E.Oracle Page Gate
The prototype evaluates the page-policy interface with an  
oracle observing the manual page score. At threshold τ, resolved 
pages with Spage(q,u)  ≥ τ are withheld; unresolved pages go to  
review. RAGAS separates context relevance from faithfulness,  
motivating distinct retention and residual-risk measures [5]. We  
also require at least three retained pages as a fixed evidence-
sufficiency proxy. The resulting curve defines a risk–coverage  
target  for  learned  page  models.  Table  I  summarizes  each  
component and its claim boundary.
The  interface  contains  four  actions:  pass,  contextualize,  
exclude, and review. Pass retains a page without certifying it  
unbiased; contextualize retains evidence while disclosing the  
relationship; exclude withholds evidence when neutral synthesis  
is inappropriate; and review handles missing or insufficient  
provenance. The oracle implements pass, exclude, and review;  
contextualization  requires  answer  generation  and  remains  
outside the current evaluation.
TABLE I. EVIDENCE ROLES IN THE EVALUATED PROTOTYPE
ComponentRecordsEvaluated purposeUnsupported inference
Manual audit200 URLsCalibration and missingnessAnnotation reliability or prevalence
Scenario20,000 rowsWorkload and threshold stressLive-Web behavior
Query router4,800 formsControlled mutation recoveryDeployed-query accuracy
Oracle gate200 URLsRisk–coverage targetClassifier accuracy or answer effect
IV. RESULTS
A.Manual Page and Query Units
Among the 200 manual records, 186 have resolved scores,  
and 72 of those 186 (38.7%) meet the 0.50 threshold. Mean  
scores are 0.477 for PROD, 0.418 for SOFT, 0.291 for POL, and  
0.248 for YMYL. PROD and SOFT each have a 55.3% high-
page share; the corresponding shares are 23.9% for POL and  
19.6% for YMYL. The cluster-bootstrap intervals in Fig. 2  
overlap, and each vertical contains only five audit queries. These  
values  therefore  serve  as  calibration  anchors  rather  than  
population estimates.
Fig. 3 illustrates the effect of manual anchoring on activated-
family  counts  at  the  main  threshold.  Rank-discounted  
aggregation  yields  a  different  family-level  pattern.  Fig.  4  
presents  the  highest-scoring  anchored  constructed  query  
families. In the  manual audit, by contrast, only PROD-003  
(0.614) and PROD-001 (0.560) activate at τ = 0.50. No SOFT 
audit query activates, even though SOFT has the same page-
level high-score share as PROD. This contrast illustrates why  
family activation depends on aggregate exposure rather than raw  
page counts alone across query families.
B.Calibration Effect and Workload
Table  II  reports  the  manual-anchored  scenario  statistics  
across verticals. Fig. 5 illustrates the calibration shifts and their  
effect on family activation. The fitted intercepts are −0.025 for  
PROD, −0.049 for SOFT, −0.056 for POL, and −0.012 for  
YMYL. All four lower the raw scenario scores because the  
unadjusted means exceed the manual anchors. At  τ = 0.50, calibration  reduces  activated  families  from  119  to  105:  45  
PROD, 33 SOFT, 18 POL, and 9 YMYL. SOFT shows the  
largest change, falling from 41 to 33 activated families.
Fig. 2.Single-coded manual real-URL pilot. Error bars are 95% query-cluster  bootstrap intervals from five query families per vertical.
Fig. 3.Manual anchoring and activated-family counts at τ = 0.50.
Fig. 4.Highest-scoring anchored canonical families.

TABLE II. MANUAL-ANCHORED SCENARIO BY VERTICAL
VerticalRaw meanδᵥAnchored meanActivated
PROD0.497−0.0250.47745/100
SOFT0.453−0.0490.41833/100
POL0.324−0.0560.29118/100
YMYL0.253−0.0120.2489/100
Fig. 5.Calibration shifts and their effect on family activation. Anchored means  equal manual means by construction.
Table III reports threshold sensitivity: 163, 128, 105, 90, and  
67  families  activate  at  0.40,  0.45,  0.50,  0.55,  and  0.60,  
respectively. At 0.50, activation is 105/400 (26.3%) in a panel  
with equal vertical shares and 50 synthetic ranks per query. This  
is conditional workload, not deployed review load or live-query  
prevalence. Query mix, ranking rules, publisher concentration,  
and retrieval depth may change activation; these effects require  
validation with actual retrieved sources.
TABLE III. ANCHORED THRESHOLD SENSITIVITY
τActivatedAudit activatedEvidence retained
0.40163/4007/200.459
0.45128/4004/20–
0.50105/4002/200.617
0.5590/4002/20–
0.6067/4001/200.795
C.Family-Held-Out Router
As  expected  under  family-held-out  evaluation,  exact  
canonical matching has zero recall on unseen mutations. The  
risk lexicon reaches 0.401 precision, 0.625 recall, and 0.489 F1.  
Word TF–IDF logistic regression reaches 0.521 F1. The word-
plus-character TF–IDF model reaches 0.546 precision, 0.492  
recall, 0.518 F1, and 0.748 accuracy. The two learned routers  
therefore have similar F1 at this scale, while the word-plus-
character model favors precision over recall.
These results support a limited engineering claim: simple  
text models recover part of the scenario-derived activation rule  
under unseen controlled reformulations. A missed activation can  
bypass provenance analysis, whereas a false activation increases  
review workload. Future tuning should therefore report both  
activation recall and activation load. Validation still requires  
independent labels across natural query variants.D.Oracle Risk–Coverage Target
Fig. 6 shows the oracle page-gate risk–coverage target under  
different  thresholds.  Before  gating,  mean  rank-discounted  
manual score mass is 0.346. At τ = 0.50, the oracle retains 61.7%  
of discounted evidence and leaves residual score mass 0.099.  
Nineteen of 20 queries meet the three-page evidence-sufficiency  
proxy, with 5.7 retained pages on average. At 0.40, evidence  
retention drops to 45.9%, and 17 queries meet the proxy. At 0.60,  
retention rises to 79.5%, all 20 queries meet the proxy, and  
residual score mass increases to 0.195.
The reduction at  τ = 0.50 is mechanically coupled to the  
score observed by the oracle. ARES likewise separates context  
relevance, answer faithfulness, and answer relevance [6]. Our  
result is neither classifier accuracy nor an answer-quality effect;  
it defines a retained-evidence level for comparing a future page  
model against residual manually scored mass.
Fig. 6.Oracle  page-gate  target.  Marker  size  gives  the  share  of  queries  satisfying the retention proxy of three pages.
V. TRIAGE AND EVALUATION CONTRACT
A.Runtime Path
Seven reproducible modules form the evaluated prototype: a  
frozen scenario-query panel, ranked scenario generation, manual  
vertical  calibration,  rank-discounted  aggregation,  controlled  
mutation, a family-held-out router, and oracle policy evaluation.  
Because content edits can alter visibility in generative engines  
[7], exported rows retain page provenance, origin, family, rank,  
raw  and  calibrated  scores,  and  threshold  decisions  for  
regenerating workload and router results.
At runtime, the trained family router receives query text and  
returns an activation decision. Only activated queries request  
deeper query-page provenance analysis. Retrieved documents  
can  steer  generated  answers  under  PoisonedRAG  [8],  so  
provenance review remains a runtime concern rather than a  
publisher  label.  Unactivated  queries  still  undergo  ordinary  
relevance  and  citation-support  checks;  routing  prioritizes  
provenance review while leaving those checks intact.
B.Page Policy
A learned model should replace the oracle only if it uses  
query-page  evidence  rather  than  domain  reputation  alone.  

Relevant  inputs  include  ownership,  sponsorship,  redirects,  
advocacy  framing,  evaluated  passages,  and  uncertainty;  
affiliate-abuse measurements show why redirects and tracking  
are evidence-bearing features [9]. Evaluation should hold out  
canonical families and domains so memorized publishers or  
disclosure templates cannot masquerade as generalization.
Filtering  is  only  one  policy  action;  disclosure  studies  
document inconsistent affiliate labeling on social platforms [10].  
A retained set may overrepresent one publisher or omit unique  
evidence, while a high-scoring page may still contain supported  
information. Selection should balance risk, relevance, support,  
and publisher diversity under an evidence-sufficiency constraint.  
Contextualization can preserve useful material with relationship  
disclosure beside each affected claim.
C.Downstream Test
An  answer-level  effectiveness  claim  requires  matched  
generation from the same archived candidate set under control,  
filter, and contextualize conditions. Because AdIntuition infers  
endorsement disclosure from content [11], the primary endpoint  
is whether contextualization preserves relationship information  
near  the  affected  claim.  Model  version,  prompt,  decoding,  
source  order,  and  evidence  budget  should  be  fixed,  and  
evaluators  should  remain  blind  to  condition.  Support,  
answerability, abstention, diversity, hallucination, and latency  
provide complementary measures.
No matched answers or answer ratings are included in the  
present study. The downstream experiment is specified so that a  
later implementation can test the interface without attributing  
answer-level improvement to the current evidence.
The same archived candidates should feed every condition.  
Rerunning retrieval after filtering would confound the policy  
action with changes in rank or page availability. Each answer  
record  should  preserve  candidate  order,  retained  passages,  
exclusions,  disclosures,  abstention,  and  the  final  evidence  
budget. This design allows failures to be localized to retrieval,  
page assessment, selection, or generation rather than assigning  
every error to the triage layer.
VI. DISCUSSION AND LIMITATIONS
The  page-  and  family-level  results  support  keeping  
activation and page treatment separate. Coordination patterns  
used to detect political astroturfing also show why publisher-
level labels can be too coarse [12]. PROD and SOFT have the  
same high-page share in the manual audit, yet only PROD  
contains  activated audit queries. In the  scenario, calibration  
changes  workload  across  verticals,  while  family-held-out  
routing provides a stronger test than exact matching.
The oracle curve exposes the central trade-off: reducing  
residual relationship-risk mass also removes evidence. Affiliate  
fraud can remain invisible in a quoted passage [13], so the page  
score  determines  review  priority  rather  than  deception  or  
factuality. Evaluation should pair risk reduction with retention,  
the  three-page  evidence-sufficiency  proxy,  and  publisher  
diversity. Missing evidence goes to review rather than being  
treated as safe.Manual anchoring transfers four vertical means, not the joint  
distribution of real queries, publishers, and ranks. Synthetic  
identifiers  and  fixed  rank  terms  omit  retrieval-dependent  
publisher concentration, duplicate sources, temporal changes,  
and access failures. These omissions can affect activation and  
retained  evidence.  Thus,  20,000  rows  support  controlled  
workload  analysis,  not  independent  validation.  The  26.3%  
activation is conditional on the generator; Section VII specifies  
an archived-retrieval check to assess transfer to actual retrieval.
RAGTruth labels unsupported RAG output at word level  
[14], whereas our provenance score asks whether a supported  
passage comes from a materially related source. Relevance,  
factual support, and provenance therefore remain separate: each  
governs  passage  utility,  claim  support,  or  relationship  
disclosure. This distinction matters especially when exclusion  
would remove unique evidence.
The  single-coded  pilot  establishes  neither  annotation  
reliability  nor  codebook  replicability.  The  query-cluster  
intervals  in  Fig.  2  reflect  variation  across  sampled  queries  
conditional on recorded ratings; they do not measure inter-rater  
uncertainty. As these ratings set the four calibration means,  
annotation differences could propagate to scenario activation.  
Section VII specifies a double-rating audit and a check of its  
effects on the anchors. These prospective analyses have not been  
conducted.
The  corpus  lacks  search-surface  metadata,  locale,  
timestamped snapshots, archive URIs, and content hashes. Five  
queries per vertical cannot support prevalence estimates or exact  
retrieval replication. Scenario rows contain no live rankings, real  
publishers, automated reviewer calls, or human query labels.  
Router metrics use the scenario activation rule and one family  
split;  split-to-split  variation  is  unknown.  No  learned  page  
classifier, matched RAG generation, answer ratings, or serving  
latency are evaluated. Conclusions remain limited to pilot score  
behavior,  workload  under  the  generator,  controlled-
reformulation routing, and oracle risk–coverage analysis.
VII. AUDITABILITY AND CLAIM GATE
Table IV defines the claim boundary. The analysis retains  
raw and calibrated scores, seed 20260716, and primary threshold  
τ = 0.50. Unresolved rows remain missing; stored geometric  
means  agree  within  0.00046  of  rounded  values.  Table  V  
summarizes the next-stage tests. The following protocols are  
prospective, not additional results.
TABLE IV. CLAIM-TO-EVIDENCE GATE
ResultSupported claim Unsupported claim
Manual anchorsCalibration targets Reliability or prevalence
ScenarioBehavior under generator Live-Web rate
Router testControlled mutation recovery Deployed-query accuracy
Oracle gateRisk–coverage target Learned classifier effect
RAG protocolEvaluation protocol Causal answer improvement
Runtime records should preserve query/family identifiers,  
URL, rank, evidence, dimension scores, uncertainty, and policy  
action. Failed access requires review or replacement, never an  
automatic pass.

TABLE V. REQUIRED NEXT-STAGE EVALUATION
ComponentHeld-out unit / outcome Utility constraint
AnnotationOverlap / weighted agreement Evidence sufficiency
Query routerFamily / macro-F1 and recall Activation workload
Page classifierFamily/domain / action P/R Abstention and access
RAG answerPaired query / disclosure Support and latency
For  each  policy  action,  the  record  should  identify  the  
supporting passage or ownership evidence and preserve the  
original rank. This would let a later audit trace an activation  
decision back to the pages that contributed to the family score.  
Recording retained and withheld pages together would also  
expose  cases  where  exclusion  removes  the  only  available  
support for a claim.
An overlap audit  would use  40 resolved pages, ten per  
vertical, spanning reviewers, queries, and both sides of 0.50  
where  available.  Two  raters  independently  score  identical  
archived evidence with a frozen codebook, blinded to prior and  
each  other’s  ratings.  Before  adjudication,  report  linearly  
weighted Cohen’s  κ per dimension, unweighted  κ at the 0.50 
composite  threshold,  raw  agreement,  and  query-cluster  
bootstrap intervals. Report stratum counts and unresolved pages  
separately; then check adjudication’s effects on anchors and  
activation.
The overlap audit should retain both original ratings before  
adjudication. Reporting disagreement by dimension would show  
whether uncertainty concerns the source relationship, disclosure,  
or apparent independence. Recomputing the anchors from the  
paired ratings would then test how that uncertainty changes  
family  activation  under  the  same  frozen  scenario,  without  
treating adjudicated scores as independent validation.
For retrieval, archive ranked URLs and content for new  
natural queries in all four verticals. Record retriever settings,  
locale, time, ranks, publishers, access status, and hashes. Fix  
calibration on disjoint development data. On held-out families,  
compare activation, retention, and publisher diversity at matched  
top-k; repeat snapshots to assess temporal change. Report each  
vertical with query-cluster intervals and interpret transfer only  
for that retrieval setting.
The archived-retrieval comparison should report the number  
of candidate pages available before each policy action. Keeping  
those candidates fixed would separate the effect of the triage rule  
from changes in retrieval. Comparisons across snapshots would  
then  address  temporal  instability,  while  reporting  publisher  
diversity alongside retained evidence would make concentration  
visible.
VIII. CONCLUSION
Query-aware triage separates family activation from page  
treatment. The single-coded pilot supplies provisional anchors;  
the calibrated 20,000-row synthetic scenario supports controlled  
aggregation, reformulation, routing, and workload analysis. The  
oracle curve defines a risk–coverage target, not a measured  
classifier  benefit.  Aspect-attentive  sponsorship  detection  
motivates a later page model [15], without validating it here.  
Reliability and transfer to real retrieval remain open; we specify  
double-rating and archived-retrieval protocols to address them.  
Present activation rates are conditional on the generator and do  not estimate deployed review workload, live-Web prevalence, or  
downstream answer-quality gains.
ACKNOWLEDGMENT
This research received no external funding. The authors  
declare no conflicts of interest. GPT-5.6 was used to polish the  
language and correct grammar.
REFERENCES
[1]N. F. Liu, T. Zhang, and P. Liang, “Evaluating verifiability in generative  search  engines,”  in  Findings  of  the  Association  for  Computational  Linguistics:  EMNLP  2023,  2023,  pp.  7001–7025,  doi:  10.18653/v1/2023.findings-emnlp.467.
[2]T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to  generate text with citations,” in Proc. EMNLP, 2023, pp. 6465–6488, doi:  10.18653/v1/2023.emnlp-main.398.
[3]F. Petroni et al., “Improving Wikipedia verifiability with AI,” Nature  Machine Intelligence, vol. 5, pp. 1142–1148, 2023, doi: 10.1038/s42256-023-00726-1.
[4]N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and  P. Liang, “Lost in the middle: How language models use long contexts,”  Transactions of the Association for Computational Linguistics, vol. 12,  pp. 157–173, 2024, doi: 10.1162/tacl_a_00638.
[5]S.  Es,  J.  James,  L.  Espinosa  Anke,  and  S.  Schockaert,  “RAGAs:  Automated evaluation of retrieval augmented generation,” in Proc. EACL  System Demonstrations, 2024, pp. 150–158, doi: 10.18653/v1/2024.eacl-demo.16.
[6]J.  Saad-Falcon,  O.  Khattab,  C.  Potts,  and  M.  Zaharia,  “ARES:  An  automated  evaluation  framework  for  retrieval-augmented  generation  systems,”  in  Proc.  NAACL,  2024,  pp.  338–354,  doi:  10.18653/v1/2024.naacl-long.20.
[7]P. Aggarwal, V. Murahari, T. Rajpurohit, A. Kalyan, K. Narasimhan, and  A. Deshpande, “GEO: Generative engine optimization,” in Proc. 30th  ACM SIGKDD Conf. Knowledge Discovery and Data Mining, 2024, pp.  5–16, doi: 10.1145/3637528.3671900.
[8]W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge  corruption attacks to retrieval-augmented generation of large language  models,” in 34th USENIX Security Symposium, 2025, pp. 3827–3844.
[9]N.  Chachra,  S.  Savage,  and  G.  M.  Voelker,  “Affiliate  crookies:  Characterizing  affiliate  marketing  abuse,”  in  Proc.  ACM  Internet  Measurement Conf., 2015, pp. 41–47, doi: 10.1145/2815675.2815720.
[10]A. Mathur, A. Narayanan, and M. Chetty, “Endorsements on social media:  An empirical study of affiliate marketing disclosures on YouTube and  Pinterest,” Proceedings of the ACM on Human-Computer Interaction,  vol. 2, no. CSCW, art. 119, 2018, doi: 10.1145/3274388.
[11]M.  Swart,  Y.  Lopez,  A.  Mathur,  and  M.  Chetty,  “Is  this  an  ad?:  Automatically  disclosing  online  endorsements  on  YouTube  with  AdIntuition,” in Proc. CHI Conf. Human Factors in Computing Systems,  2020, doi: 10.1145/3313831.3376178.
[12]D. Schoch, F. B. Keller, S. Stier, and J. Yang, “Coordination patterns  reveal online political astroturfing across the world,” Scientific Reports,  vol. 12, art. 4572, 2022, doi: 10.1038/s41598-022-08404-9.
[13]P. Snyder and C. Kanich, “Characterizing fraud and its ramifications in  affiliate marketing networks,” Journal of Cybersecurity, vol. 2, no. 1, pp.  71–81, 2016, doi: 10.1093/cybsec/tyw006.
[14]C. Niu, Y. Wu, J. Zhu, S. Xu, K. Shum, R. Zhong, J. Song, and T. Zhang,  “RAGTruth:  A  hallucination  corpus  for  developing  trustworthy  retrieval-augmented language models,” in Proc. ACL, 2024, pp. 10862–10878, doi: 10.18653/v1/2024.acl-long.585.
[15]S.  Kim,  J.-Y.  Jiang,  and  W.  Wang,  “Discovering  undisclosed  paid  partnership on social media via aspect-attentive sponsored post learning,”  in  Proc.  ACM  WSDM,  2021,  pp.  319–327,  doi:  10.1145/3437963.3441803.