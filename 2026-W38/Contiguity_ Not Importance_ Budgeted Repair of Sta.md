# Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits

**Authors**: Mingyang Mao, Wyatt Mackey, Xiaomin Lin

**Published**: 2026-09-16 01:13:04

**PDF URL**: [https://arxiv.org/pdf/2609.17983v1](https://arxiv.org/pdf/2609.17983v1)

## Abstract
KV-cache reuse can reduce inference cost in retrieval-augmented generation and agentic systems, but cached contexts may become stale when retrieved knowledge, working memory, or user state is edited. Under causal self-attention, even a local edit can affect downstream KV states. A full re-prefill reliably restores consistency but is costly, whereas refreshing only the edited span can leave downstream dependencies stale. We formulate in-place repair as budgeted recomputation and compare training-free position-selection policies on a factual RAG benchmark with matched direct and derived edits. Across three model families, all policies repair direct cases, but derived cases clearly separate them. At the primary budget, a contiguous edit-local window recovers at least 0.94 of the post-edit answer margin and substantially outperforms attention-based, KV-deviation, and structural selectors. Mechanistic analysis shows that position sets effective under clean-state transplantation can fail under actual recomputation because scattered positions inherit surrounding staleness. The edit-local advantage also depends on adjacency and largely disappears when the answer-bearing text moves downstream. Because answer-relevant edits almost always corrupt model behavior, failure severity is difficult to predict, and repair is 13-21 times faster than full re-prefill, our results support unconditional edit-local repair when the dependent text remains adjacent to the edit.

## Full Text


<!-- PDF content starts -->

Contiguity,NotImportance:BudgetedRepairofStaleKVCachesAfterDocument
Edits
Mingyang Mao1, Wyatt Mackey2, Xiaomin Lin1∗
1University of South Florida, Electrical and Computer Engineering
2DEVCOM Army Research Laboratory
{mmao, xlin2}@usf.edu, wyatt.t.mackey.civ@army.mil
Abstract
KV-cache reuse can reduce inference cost in retrieval-
augmented generation and agentic systems, but cached con-
texts may become stale when retrieved knowledge, working
memory, or user state is edited. Under causal self-attention,
even a local edit can affect downstream KV states. A full
re-prefill reliably restores consistency but is costly, whereas
refreshing only the edited span can leave downstream depen-
denciesstale.Weformulatein-placerepairasbudgetedrecom-
putationandcomparetraining-freeposition-selectionpolicies
onafactualRAGbenchmarkwithmatcheddirectandderived
edits. Across three model families, all policies repair direct
cases, but derived cases clearly separate them. At the pri-
marybudget,acontiguousedit-localwindowrecoversatleast
0.94 of the post-edit answer margin and substantially outper-
formsattention-based,KV-deviation,andstructuralselectors.
Mechanistic analysis shows that position sets effective under
clean-statetransplantationcanfailunderactualrecomputation
becausescatteredpositionsinheritsurroundingstaleness.The
edit-local advantage also depends on adjacency and largely
disappears when the answer-bearing text moves downstream.
Because answer-relevant edits almost always corrupt model
behavior, failure severity is difficult to predict, and repair is
13–21×faster than full re-prefill, our results support uncon-
ditional edit-local repair when the dependent text remains
adjacent to the edit.
Introduction
RAGpipelinesandLLM-basedagentsincreasinglyreuseKV
states for retrieved documents or memory blocks across re-
quests, avoiding repeated prefill (Yao et al. 2025; Bergman
et al. 2025; Ye et al. 2026; Pan et al. 2026). This optimiza-
tion assumes cached source content does not change. In de-
ployment, that assumption breaks when a factual record is
corrected (Ouyang et al. 2025; Cohen et al. 2024), a policy
isrevised,oranagent’sworkingmemoryisupdated(Packer
et al. 2023). The source text then carries the update, while
retainedKVtensorsencodetheolderversion.Arequestcan
therefore be answered from a representation that no longer
matches the knowledge base. Existing document-level reuse
methods optimize reuse and composition, but do not syn-
chronize cached representations after a source edit.
Repairingthismismatchisnotconfinedtotheeditedspan.
Continuing to serve from a stale cache can reduce response
accuracy (Ouyang et al. 2025), and causal self-attention
∗Corresponding author.
valid edited stale (attended to old text)(a) In -place edit stales the KV cache
(b) Benchmark construction: direct vs derived
Direct probe:  answer span =
[Entity Record]
The registry value for X is D668 J651
Derived probe:  answer span ≠ edit span 
[Alias Record]
X uses alias A Bedit span 
[Lookup Table]
A has value 4471
B has value 8730 Answer span unedited
Q: What value does X’s alias map to?
A: 8730    without repair:   4471D tokens downstreamCached support -policy document
Plan: Enterprise Gold 
Initial response SLA: 24 hours edited field 
Data retention: 90 days 
Monthly API quota: 10 million requests 
Escalation channel: Priority Support
edit spanFigure 1:Problem and probe design.(a) An in-place edit
to a cached document leaves the edited span’s KV entries
encoding old text and all downstream entries stale. (b) Each
benchmark item pairs the same question with two context
structures:adirectprobe,wheretheeditrewritestheanswer
text,andaderivedprobe,wheretheanswersitsinanunedited
lookup entry downstream of the edited alias. Without repair
the cache returns the pre-edit answer.
leaveseverydownstreamKVstatedependentontheoldcon-
tent. A full re-prefill restores consistency but re-encodes an
entire long context after an edit of only a few tokens. Re-
cent work has begun to repair the cache in place after such
small edits, but each proposed method carries a limitation.
KVEraserreplacesatargetspanwithlearnedsteeringstates,
yet requires per-model training and targets deletion rather
arXiv:2609.17983v1  [cs.AI]  16 Sep 2026

than factual replacement(Li et al. 2026). MTN recomputes
downstreamnotesrankedbytheircausaleffect,usinganora-
clesignalincontrolledagenttasks(Li2026).Neitheranswers
the matched-budget question for factual RAG under actual
recomputation. To build a low-budget, training-free repair
policy, we therefore first ask which downstream positions
matter most once the edited span itself has been refreshed.
Ourstudypairsfactualeditswithcontextsofroughly5000
retrievedtokens.Eachquestioneitherasksfortheeditedfact
directlyorrequiresatwo-hopderivationthroughunchanged
downstream text. Starting from the stale cache, every policy
refreshes the known edit span and selectsKdownstream
positions under the same recomputation budget. We com-
pare an edit-local window, structural delimiters (Li 2026),
stale-query attention (Wang et al. 2026), CacheBlend-based
KV deviation (Yao et al. 2025), and random selection. A
transplant-derived causal ranking is included only as a non-
deployable diagnostic. Evaluation covers development and
held-out cohorts across Llama, Qwen, and Mistral, plus a
1099-itemvariantthatmovestheanswer-carryingtextdown-
stream.Weusepre-specifiedcriteriaandreportheld-outmar-
gin recovery and flip rate.
Three results define the baseline. AtK=32, every policy
saturates on direct questions, while derived answers sepa-
rate them. The edit-local window recovers0.94–1.01of the
answer margin and flips0.95–1.00of held-out items, beat-
ing every alternative by0.46–1.00margin recovery (Holm-
correctedp≤3×10−9). This advantage depends on adja-
cency. Moving the answer-carrying text250tokens down-
stream reduces edit-local recovery to0.01–0.09, with stale-
queryattentionhelpingononlyonemodelfamily.Transplant
recoverability also fails to predict repair under recomputa-
tion. The causal ranking recovers0.92–0.97of the margin
when clean states are transplanted, but the same positions
recover only0.01–0.21when recomputed atK=8. Recom-
putationisworseoneveryheld-outitem.Atransplantedstate
importsinformationfromacorrectcache,whereasarecom-
putedstatereadsstalesurroundingstates.Acontiguousedit-
localwindow succeedsby rebuildingthat dependencychain
in order. Finally, answer-relevant edits almost always break
thestalecache(baserate≥0.988),andcheapfeaturespoorly
predict failure severity (best out-of-foldρ= 0.17). Because
repairruns13–21×fasterthanre-prefill,applyingedit-local
repair unconditionally is the practical policy within this set-
ting. It is also the training-free baseline that future selection
methods must beat.
Insummary,thispapermakesthefollowingkeycontribu-
tions:
•We formulate stale-cache repair as budgeted recomputa-
tion and introduce paired direct, derived, and distance-
controlled factual edits.
•Wecomparefivetraining-freeselectorsatamatchedbud-
get across three model families, establishingEditLocal
as a strong adjacent-block baseline.
•Weseparatelocalizationunderclean-statetransplantfrom
repair under recomputation.
•Weshowthatanswer-relevanteditswarrantunconditional
repair within this benchmark because failure is frequent,severity is hard to predict, and repair is inexpensive.
Related Works
Document-level KV reuse.RAGCache stores intermedi-
atestatesofretrievedknowledge,whileTurboRAGprecom-
putesper-chunkKVcachesforreuseacrossqueries(Jinetal.
2025;Luetal.2025).Bothreducerepeatedprefillbyreusing
stored states of retrieved text. Neither addresses how those
states should be updated after the source text changes. HoH
shows that outdated retrieved evidence can reduce RAG ac-
curacyevenwhencurrentevidenceisavailable(Ouyangetal.
2025), but it studies stale information at the text level rather
thanrepairofanalreadycachedrepresentation.Westudythe
systemsproblemthatfollowsasourceedit:howmuchofone
stale document cache must be recomputed?
Selective recomputation.Selective recomputation ad-
dresses a different cache mismatch. CacheBlend restores
missing cross-attention by recomputing positions with large
KV deviations, while ProphetKV prioritizes positions us-
ing query relevance (Yao et al. 2025; Wang et al. 2026).
EPIC recomputes a small, fixed set of initial chunk tokens,
Cache-Craft repairs reusable chunk caches through limited
recomputation, and KVShare selects high-deviation states
during prefill and decoding (Hu et al. 2025; Agarwal et al.
2025; Yang et al. 2025). Across these methods, the source
chunksareunchangedandthemismatchcomesfromreusing
their caches in a new context. Our experiments transfer the
deviation-andquery-basedselectionsignalstoacachewhose
source text has changed and compare them at the same
downstream-position budget.
KV-cache editing.KV-cache editing is closest to our set-
ting. KVEraser replaces a target span with learned steering
states to erase its influence (Li et al. 2026). Li (2026) shows
thatchangingafieldcanleaveoldconclusionsindownstream
states and ranks those states by causal effect. Leyline pro-
vides serving primitives for removing or replacing cached
spans, including positional correction for length-changing
edits (Ma, Eitzinger, and Koestler 2026). We isolate length-
preserving factual replacement and ask which training-free
selector works when its chosen positions are actually re-
computed at a matched budget. This separates our setting
fromlearnederasure,cachesplicing,andpositionsetsranked
through oracle-state transplantation.
Cache Repair as Budgeted Recomputation
Question Setting.A serving system prefills a contextC
and stores its key–value states for reuse. After an edit pro-
ducesC′,thestoredcacheK(C)isstale;afullprefillK(C′)
istheoraclerepairtarget,notacompetingmethod.In-place
repairinsteadrefreshesselectedentriesofK(C)towardthis
target.
We isolate one contiguous, length-preserving edit:|C|=
|C′|=nandc i=c′
ioutside the half-open spanS=
[sstart, send). Positions and rotary phases therefore remain
fixed; length-changing edits, which also require positional
correction,areoutsideourscope(Ma,Eitzinger,andKoestler
2026).Eachbenchmarkitempairsadirectcondition,whose

answer lies inS, with aderivedcondition, whose answer
liesinunchangeddownstreamtextbutdependsontheedited
fact(Figure1b).Thepairotherwisesharesitsretrieveddoc-
uments, insertion point, subject, value pair, and query, and
we report the conditions separately.
The repair interface.Every policy uses the same op-
erator and differs only in itsKselected positions. Let
D= [s end, n−1)be the downstream candidate pool and
letA π⊆D,|A π|=K,bethepositionschosenbypolicyπ.
The repaired set isP π=S∪A π, and
eK=R 
K(C), C′, Pπ
.
The operator recomputesP πfromC′layer by layer and
leaves all other states unchanged, using the same selective-
recomputation primitive as prior systems (Yao et al. 2025).
Its full-position endpoint is
R 
K(C), C′,{0, . . . , n−1}
=K(C′).(1)
The known edit span is always recomputed and shared
across policies, while the fresh query is never cached and is
recomputed during scoring. Thus,Kcounts only additional
downstream positions. We exclude upstream positions be-
cause causal attention prevents them from depending on the
edit, and verify this invariant numerically. The question is
which deployable rule choosesA πmost effectively.
Cost accounting.We separate three costs: common work
fortheeditspanandfreshquery,thematchedrepairbudgetof
K·Ltoken-layers,andselectoroverhead.Thecommonterm
cancels in policy comparisons, while overhead is reported
outsideK. Text- and position-only rules have no selector
forward;Attentionadds a stale-cache scoring pass (Wang
etal.2026),andCacheBlendaddsroughlyoneprefilllayer
overtheprefix(Yaoetal.2025).Weusewall-clocklatencyas
the headline systems measure and token-layers as a device-
independent secondary measure; Section reports both. The
full construction, interface invariants, and cost decomposi-
tion appear in the supplementary appendix.
What counts as repaired.Answer accuracy alone cannot
distinguish caches that emit the same string with different
underlying preferences. Each item therefore defines a pre-
editanswera oldandapost-editanswera new.Forcachestate
x, letm xbe the difference between their length-averaged,
teacher-forced log probabilities. Our primary metric is
MR =mrepaired −m stale
moracle−m stale,(2)
where0denotesnochangefromthestalecacheand1reaches
theoracle.MRisunclipped,andzero-gapitemsareexcluded.
Wepairthisinternalmeasurewithvisiblebehavior.Lety b
be the greedy continuation of at most16tokens for retained
itemb. The flip rate is
Flip =1
|B|X
b∈B1
anew∈yb∧a old/∈yb
.(3)
MR captures partial recovery before the decision boundary,
whereas flip records whether generation exposes only the
new answer. We report both, with KL recovery as a sec-
ondarydistributionalmeasure.Theappendixgivesthecom-
plete scoring and edge-case rules.Policy Rule (all pickKpositions from
[send, n−1))
EditLocaltheKpositions immediately after the
span
Structuralstructural tokens, nearest to the span first
Attentiontop-Kby stale-cache attention from the
query
CacheBlendtop-Kby layer-2 key/value deviation
Randomuniform from the pool,5seeds
CausalOracletop-Kby measured causal effect (needs
oracle cache)
CarrierWindowcontiguous window at the true carrier
block (needs its location)
Table1:Selectionpolicies,frozenbeforeheld-outevaluation.
OnlyKvaries; grey rows are diagnostics.
Selection policies.Table 1 compares four deployable
signals—edit proximity, structure, stale-query attention
(Wang et al. 2026), and KV deviation (Yao et al. 2025)—
with a matched random control. These policies may use the
stalecache,editedcontext,editlocation,andquery,butnever
the oracle cache or answers. Grey rows use unavailable ora-
cleorcarrierinformationandarediagnostics,notdeployable
methods or upper bounds on recomputation.
The primary budget isK=32; curves overK∈
{8,16,32,64}aredescriptive.ThestructuralEditLocal@∂
variantinSection isexploratorybecauseitwasinferredfrom
those curves. Full policy definitions and provenance appear
in the appendix.
Transplantisnotrecomputation.Aselectedpositionset
canbeusedintwoways.TransplantcopiesitsKVstatesfrom
the oracle cache into the stale cache, while recomputation
rebuilds them within the stale cache. Only recomputation is
deployable because transplant requires the oracle cache. We
use transplant only as a localization diagnostic.
The two operators provide different information. A trans-
planted state was computed with an otherwise correct cache
and can therefore import information from outside the se-
lected set. A recomputed state must instead read the still-
stale cache around it. Positions that appear sufficient under
transplantmaythereforefailunderrecomputation.Weapply
both operators to identical position sets to measure this gap.
Experimental Setup
Pairededitbenchmark.Eachitemcontainsroughly5000
tokens of shuffled HotpotQAfullwikivalidation para-
graphs (Yang et al. 2018), used only as realistic retrieval
background; HotpotQA questions and answers are not used.
Weinsertonesyntheticrecordatadocumentboundaryabout
30%into the context.Backgrounddenotes the filler para-
graphs,whereasitemdenotesthefullconstructionandisthe
unit of every paired comparison. The direct block states the
queriedfieldandrewritesitsvalue.Thederivedblockinstead
rewritesanaliasthatselectsbetweentwodownstreamlookup
rows; both rows are present before the edit and remain un-
changed.Thus,onlythelocationoftheanswerrelativetothe

0 10 20 30 40 edit 100 1000
token offset fr om the end of the edit span(a)
EditLocal
Attention
CacheBlend
Structural
Random
CausalOraclenext document 
answer tokens1.01
0.64
0.25
0.15
0.13
0.35MR
margin recovery
transplant recompute0.00.51.0
@32
@8EditLocal@32
CausalOracle
Random@32(b)Figure2:Contiguity,notlocalization,decidesrepair.(a)Selectionsforarepresentativeheld-outLlamaitematK=32.Orange
markstheeditspan,repairedoutsidethebudget,andgreymarksstatesleftstale.OnlyEditLocalformsanunbrokenpathfrom
the edit through the answer. (b) Margin recovery under transplant and recomputation on identical sets from15development
items. Lines connect the two operators. Solid lines denote Llama and dashed lines Qwen. GreyCausalOracleresults are
mechanism diagnostics without tests.
edit differs between the conditions. All synthetic subjects,
aliases, and values are absent from the background.
Build-time filters require the old and edited prefixes to
have equal length under every tokenizer, differ in one span
of at most six tokens, and place that span between25%and
40%of the context. Direct and derived pairs must match
within3%in relative edit position. At evaluation, we retain
onlyitemsforwhichtheoracleanswerscorrectlyandastale-
to-oracle margin gap exists. These model-dependent filters
defineretention.Theappendixgivesthefullconstructionand
filter provenance.
Distance-controlled variant.The main derived block
makestheeditandanswer-carryingtextadjacent.Tovarythis
geometry,wesplititsaliasandlookuprecordsintotwodoc-
uments,leavethealiasinplace,andmovethelookuprecord
downstream.Wedefined=s carrier−send∈ {0,250,1500}.
The direct condition appears only atd=0. Atd=0, the de-
rived construction is byte-identical to the main benchmark.
The distance axis extends rather than replaces the main set-
ting; all policies, budgets, and metrics remain fixed.
Models and decoding.We use Llama-3.1-8B-Instruct,
Qwen3-8B, and Mistral-7B-Instruct-v0.3 in bfloat16 with
SDPAattention,decodinggreedilywithnochainofthought.
Qwen3 receives an empty think block so that it answers di-
rectly.Becausethethreetokenizersmapthesametexttodif-
ferent token counts, a fixedKdoes not cover identical text
spans across families. Before evaluation, a frozen harness
gate compares the repaired cached path with a full forward
at the answer position under a bfloat16 tolerance fixed after
the initial smoke test.
Pre-specifiedheld-outevaluation.Developmentuses100
items. The policy roster, metrics, budgets, and tests are pre-
specified and applied consistently across all three model
families. Held-out items use different seeds and HotpotQAindicesdisjointfromdevelopment.Acandidate-poolexpan-
sion rule is fixed in advance, preventing retention shortfalls
from being repaired by loosening filters. Retention is75di-
rectand34derivedforLlama,75ineachconditionforQwen,
and60direct and58derived for Mistral. Llama’s lower de-
rivedcountreflectsfailuretosolvesometwo-hopitemseven
from a clean prefill.
All comparisons are paired by item. We report95%per-
centile intervals from10,000bootstrap resamples. A policy
meanisresampleddirectly;acontrastisformedperitembe-
foreresamplingsothatpairingispreserved.PairedWilcoxon
testsaccompanythecontrasts.AtK=32,all15comparisons
among six policies form one Holm-corrected family within
each model and condition;CarrierWindowappears only
in the distance variant. Curves overKare descriptive and
are not tested. Win–tie–loss uses a primary MR tie band of
0.05,fixedbeforeheld-outevaluationfromthespreadofse-
mantically equivalent arms, with0.01and0.10as appendix
sensitivities.Directandderivedconditionsareneverpooled.
The appendix records the cohort and statistical details.
Results
Acrossallthreemodelfamilies,everypolicysaturatesinthe
directcondition,whichservesasasanitycheck.Thederived
conditionseparatesthemethods:EditLocalnearlyrestores
the full answer margin when the dependent text is adjacent
totheedit,butitsadvantagelargelydisappearsoncethattext
moves downstream. Transplant-ranked positions also lose
most of their value under actual recomputation. Together
with near-universal stale-cache failure and the low cost of
repair, these results support unconditional edit-local repair
within the adjacent-block setting.
The Edit-Local Window Wins
AtthefrozenK=32,EditLocalrecovers0.993,1.007,and
0.937of the margin on Llama, Qwen, and Mistral and flips

Llama-3.1-8B (n=34) Qwen3-8B (n=75) Mistral-7B (n=58)
Policy MR Flip MR Flip MR Flip
EditLocal0.993±0.008 0.971.007±0.005 1.000.937±0.032 0.95
Structural0.126±0.065 0.03 0.008±0.011 0.01 0.012±0.012 0.00
Attention0.533±0.068 0.03 0.015±0.014 0.01 0.019±0.014 0.00
CacheBlend0.238±0.080 0.03 0.076±0.030 0.03 0.012±0.012 0.00
Random0.078±0.054 0.00 0.004±0.010 0.00 0.004±0.009 0.00
CausalOracle0.409±0.064 0.00 0.187±0.037 0.09 0.096±0.035 0.05
Table 2: Held-out results for the derived condition atK=32. MR is Eq. 2, reported as the mean±the half-width of a95%
percentile interval from10,000bootstrap resamples, which is symmetric to within0.003; Flip is Eq. 3. Every paired contrast
againstEditLocalis Holm-significant within its model’s15-pair family (p≤3×10−9). The grey row is a transplant-ranked
diagnostic,notamethodoranupperboundunderrecomputation.Directrowsareomittedbecauseallpoliciesachieve0.98–1.00.
95–100%ofheld-outitems(Table2).Thebestrival,Atten-
tionon Llama, reaches0.533. No deployable rival clears
0.08on Qwen or Mistral. The greyCausalOraclereaches
only0.10–0.41. On Llama, rivals can move probability to-
ward the new answer, but none flips more than one of34
items. MR thus captures partial internal repair, while Flip
records whether generation exposes the new value.
Figure2amakesthegeometricdifferencevisible.Causa-
lOracleincludes the answer token but recovers only0.35,
andCacheBlendstopsinsidetherowholdingtheoldvalue.
OnlyEditLocalrebuilds an unbroken path from the edit
through the dependent text. It beats the oracle on all34,75,
and58held-out items, and its weakest record against any
rival is32wins,2ties, and no losses. KL recovery follows
MR at0.999,1.000, and0.943.
Qwen’s mean MR above one reflects slight overshoot be-
yond the oracle margin, which Eq. 2 leaves visible. Mistral
reaches1.002atK=64,consistentwithitstokenizerrequir-
ing more than32tokens to cover the same text.
How Far the Window Should Extend
Direct results stay near one at every budget and are omitted
from Figure 3. In the derived condition,EditLocalrises
from0.11to1.00on Llama, from0.01to1.01on Qwen,
and from0.01to1.00on Mistral asKincreases from8to
64. The transition occurs atK=32. AtK=64,Attention
reaches0.94on Llama but only0.18and0.30on Qwen and
Mistral,andeveryotherruleremainsatorbelow0.45.More
budget alone does not produce the same gain.
Thethresholdmatchestheblockgeometry,anexploratory
observationmadeafterunblindingthecurves.Fromtheedit
to the block boundary, the dependent text spans a median
of24Llama tokens (maximum26) and28Qwen tokens
(maximum30).Awindowof16tokensisincomplete,while
32covers the path. This motivatesEditLocal@∂, which
recomputes to the next document boundary with no selector
forward.ItcoincideswithK=32onLlamaandQwen,while
Mistral needs a slightly larger budget. This structural rule is
an interpretation of the curves, not a separately tested arm,
and still assumes that dependent text begins where the edit
ends.Moving the Answer Away from the Edit
Thebyte-identicald=0tierreproducesthemainresults,rul-
ing out the two-document builder as the source of the dis-
tance effect. For example, QwenEditLocalscores1.002
ratherthan1.007,andCacheBlendscores0.075ratherthan
0.076. Tests are Holm-corrected within each four-pair cell
family.
Adjacency is load-bearing. Atd≥250,EditLocalfalls
to0.01–0.09MRand0.00–0.04Flip.ItmostlytiesRandom
on Qwen and Mistral, while Llama’s remaining advantage
is only+0.07.EditLocal@∂also stops before the carrier.
Attentionretains useful signal only on Llama, reaching
0.39atd=250and0.84atK=64. It stays at or below0.11
onQwenandMistral.Themainwinisthereforeanadjacency
effect, not a general advantage of the selector.
Search is only part of the problem. Even with the true lo-
cation,CarrierWindowrecovers just0.43–0.74on Llama
and Mistral, compared with0.89–0.95on Qwen. The probe
still moves the dependency as intended:61–71%of the ora-
cle’spositiveeffectmassliesinthecarrierblock,andatmost
3%remains in the edited block. Recovery from a correctly
placed window is thus model-dependent.
Llama retains only118solvable items atd=250, about
one third of the pool, so its distance cells cover only solv-
able examples. Distance also changes absolute position and
recency, which partly confounds the attention result.
Transplant Recoverability Does Not Imply
Recompute Repairability
Thegreyoraclerankspositionsbycausaleffectundertrans-
plant, but the same sets behave differently under recompu-
tation (Figure 2b). On15development items, transplant re-
covers0.93–0.99ofthemargin,whilerecomputationreaches
only0.03–0.41.Thegapisspecifictoscatteredsets:EditLo-
calandRandomdifferbylessthan0.02betweenoperators,
and all direct cells differ by less than0.003.
Held-out results repeat the gap on all three families. At
K=8,transplantrecovers0.918,0.952,and0.969onLlama,
Qwen, and Mistral, compared with0.209,0.011, and0.008
underrecomputation.Recomputationisworseoneveryheld-
out item, and all167recorded position sets match when
replayed. Even atK=32, recomputation reaches only0.10–
0.41.

8 16 32 640.000.250.500.751.00margin recovery
Llama-3.1-8B
8 16 32 64
Qwen3-8B
8 16 32 64
Mistral-7B
K (extra recomputed downstream positions)
EditLocal Structural Attention CacheBlend Random CausalOracleFigure 3: Margin recovery against the budgetKon the held-out set, derived condition. Budgets double along a logarithmic
axis, and bands are95%bootstrap intervals. The direct condition is omitted because every policy lies between0.98and1.00
fromK=8on.Thecurvesaredescriptive,andthespecificationreservestestingforK=32.EditLocalisthreshold-shapedand
saturates once the window covers the remainder of the injected block. The grey dashed line is the transplant-ranked oracle, a
diagnostic rather than a method.
Llama Qwen Mistral
Policyd=0 250d=0 250 1500d=0 250 1500
EditLocal1.00 0.09 1.00 0.01 0.01 0.94 0.01 0.02
Structural0.08 0.08 0.01 0.00 0.01 0.01 0.01 0.02
Attention0.51 0.39 0.01 0.02 0.05 0.01 0.02 0.11
CacheBlend0.19 0.08 0.09 0.01 0.01 0.01 0.01 0.02
Random0.04 0.02 0.00 0.00 0.00−0.00−0.01−0.00
CausalOracle0.35 0.42 0.22 0.22 0.22 0.09 0.10 0.13
CarrierWindow0.61 0.43 0.95 0.90 0.89 0.74 0.64 0.60
Table3:MeanMRatK=32asthelookuptablemovesdtokensdownstream.Samplesizesare191/118(Llama),372/359/354
(Qwen), and343/216/221(Mistral). Bootstrap intervals are omitted. Every half-width is at most0.06. The post-hoc Llama
d=1500cell is excluded. Grey rows are diagnostics, andCarrierWindowreceives the true carrier location.
Atransplantedstatecomesfromafullycorrectcacheand
can import information from outside the chosen set. A re-
computed state reads the stale cache around it, so scattered
positionsinheritstaleness.Acontiguouswindowinsteadre-
builds the forward chain in order. Transplant recoverability
therefore shows where clean information can act, not what
sparserecomputationcanreconstruct.Developmentdiagnos-
tics support this account:Structuralomits content tokens
carrying most positive effect mass, while stale-query atten-
tion ranks them too deep to fit the budget.
Repair Should Be Unconditional
WetestedcheaprepairtriggersonLlamaandQwenwithout
retention filters (n=175per condition). After an answer-
relevant edit, the stale cache fails on at least98.8%of items
per condition, reaches100%in both direct conditions, and
neverfallsbelow97.3%inanysplit.Thisresultislimitedto
editsthattouchtheanswerchain,butwithinthatscopethere
is little for a gate to separate.
Severity is no easier to predict. Ridge and random-forest
models using text, position, edit, and embedding features
reach a best out-of-fold Spearmanρ= 0.17(R2= 0.05)
under grouped five-fold cross-validation. Edit type distin-Model Ctx Full prefill Repair@32Speedup
Llama-3.1-8B 4096 382.8ms 27.8ms 13.8×
8192 818.1ms 39.6ms 20.7×
Qwen3-8B 4096 412.1ms 31.6ms 13.0×
8192 884.1ms 44.9ms 19.7×
Mistral-7B 4096 374.3ms 28.0ms 13.4×
8192 802.3ms 39.8ms 20.2×
Table 4: Wall-clock cost of repair (span plusK=32down-
stream positions, all layers) against a full re-prefill of the
editedprefix.Medianof20timedrunsafter3warmupruns,
on one RTX 5090, bfloat16, batch1. Both sides exclude the
fresh-queryforward,whichisidenticalforeverymethod.Re-
pair includes a defensive cache copy (<2ms), and in-place
repair is marginally faster.
guishesdirectfromderivedcases,buttheremainingfeatures
do not support a useful severity gate. Near-certain failure,
weak predictability, and low repair cost favor unconditional
repair.

What Repair Costs
At roughly5,000tokens, repair takes31–35ms versus458–
502ms for re-prefill, about15×faster on all three models
(Table 4). The ratio reaches about20×at8K. RaisingK
from8to64addsonly1–4ms,makingastructuralboundary
such asEditLocal@∂inexpensive. The token-layer proxy
gives117×–234×butoverstatesthemeasuredgain.Selector
overhead also favors local rules, which require no scoring
forward, unlikeAttentionandCacheBlend.
Sanitychecks.Nofrozenguardfiredontheheld-outbatch.
Stale and oracle outputs reproduced byte-equal generations
with margins within10−3, and all oracle rankings replayed
without mismatches. The endpoint identity in Eq. 1 passes
onLlama.Qwenpassesinmargin(0.997–1.013)butexceeds
the frozen value-tensor L2 tolerance. A null-content con-
trol places the discrepancy within bfloat16 numerical noise.
We retain the failed label. On Mistral, three of80bfloat16
smokestateschangedargmax,andallthreematchedexactly
in float32.
Discussion
Whattodeploy.Foradirectquestion,refreshingtheedited
span completes the repair. For an answer derived through
adjacent text, recompute through the end of the block. This
rule reads only text and position, and its forward is13–
21×cheaperthanre-prefill.Astructuralstoppingpointsuch
asEditLocal@∂avoids tuningK, and repair should be
unconditionalbecausefailureisnearlycertainwhileseverity
is hard to predict. This recommendation applies when the
answerdependencyremainsintheeditedblock.Beyondthat
boundary,asystemneedsadifferentrepairoperator,notonly
a larger fixed budget.
Coverage, not importance.Chunk-composition methods
repairstatescomputedfromcorrectinputsbutmissingcross-
chunkattention,solargeKVdeviationscanbeusefulsignals
(Yaoetal.2025).Aneditinsideacacheddocumentcreatesa
differentdefect:abrokendependencychain.Scatteredhigh-
scoring positions are recomputed from stale inputs, while
a contiguous window rebuilds the path in order. Selection
quality is therefore a property of the set, not each position
in isolation. Deviation and attention scores can help when
surrounding states are sound. When they are stale, repair
must cover the path that produces the answer.
Localization evidence overstates what repair can do.
Cache-editing work can rank positions by patching clean
states into a corrupted run (Li 2026), but that ranking need
nottransfertorecomputation.Transplantimportsstatespro-
duced in a correct context, including information from out-
sidethechosenset.Recomputationmustrebuildthosestates
from the stale cache. The operator gap shows that a rule
intended for recomputation must be tested under recompu-
tation.Transplantlocateswherecleaninformationcaninflu-
ence the output, but it is not an upper bound on deployable
repair.
Where repair still fails.Once dependent text moves250
tokens away,EditLocalcollapses and only stale-query at-
tention on Llama provides a useful deployable alternative.Search is not the whole problem:CarrierWindowknows
the carrier location yet remains far below full recovery on
Llama and Mistral. A second pass or a window spanning
the edit-to-carrier path may restore the missing inputs, but
neitheristestedhere.Qwen’sstrongerresultalsoshowsthat
the boundary is model-dependent.
Limitations.Edits are single, contiguous, and length-
preserving,andthesyntheticblocksmakeeveryeditanswer-
relevant. The failure rate therefore does not extend to arbi-
trary edits, and the two-value metrics do not measure long
free-form answers. Llama retains34of75held-out derived
itemsandaboutonethirdatd=250,sothosecellscoveronly
solvable examples, and distance also changes recency. The
transplantpaneluses15developmentitemswithoutMistral,
althoughtheoperatorgapisalsomeasuredonheld-outdata.
EditLocal@∂was inferred from unblinded curves rather
than tested as a registered arm. The models are dense and
near8B,andtimingusesoneGPUatbatch1whileexcluding
the identical query forward. Multi-edit and length-changing
updates, larger or sparse models, and production serving re-
main open.
Conclusion
We framed in-place repair of a stale KV cache as budgeted
recomputationandcomparedtraining-freeselectionpolicies
atamatchedbudgetonpairedfactualeditsacrossthreemodel
families. Recomputing a contiguous window from the edit
to the end of its block restores post-edit behavior almost
completely wherever the dependent text is adjacent, beats
every signal-based rule by a wide margin, and runs13–21×
fasterthanare-prefill.Theresultcarriesamechanismanda
boundary.Repairworksbyrebuildingaforwarddependency
chain, so position sets that look sufficient under transplant
failunderrecomputation,andthesamewindowfailsoncethe
chain grows long. Edit-local recomputation is the baseline
that any future policy for this problem has to beat.
References
Agarwal,S.;Sundaresan,S.;Mitra,S.;Mahapatra,D.;Gupta,
A.; Sharma, R.; Kapu, N. J.; Yu, T.; and Saini, S. 2025.
Cache-craft: Managing chunk-caches for efficient retrieval-
augmentedgeneration.Proceedings of the ACM on Manage-
ment of Data, 3(3): 1–28.
Bergman, S.; Kermarrec, A.-M.; Petrescu, D.; Pires, R.;
Randl, M.; De Vos, M.; and Zhang, J. 2025. Leveraging
approximate caching for faster retrieval-augmented genera-
tion. InProceedings of the 26th International Middleware
Conference, 340–353.
Cohen,R.;Biran,E.;Yoran,O.;Globerson,A.;andGeva,M.
2024. Evaluating the Ripple Effects of Knowledge Editing
in Language Models.Transactions of the Association for
Computational Linguistics, 12.
Hu,J.;Huang,W.;Wang,W.;Wang,H.;Hu,T.;Qin,Z.;Feng,
H.; Chen, X.; Shan, Y.; and Xie, T. 2025. EPIC: Efficient
Position-Independent Caching for Serving Large Language
Models. InInternational Conference on Machine Learning,
24391–24402. PMLR.

Jin, C.; Zhang, Z.; Jiang, X.; Liu, F.; Liu, S.; Liu, X.; and
Jin, X. 2025. Ragcache: Efficient knowledge caching for
retrieval-augmentedgeneration.ACM Transactions on Com-
puter Systems, 44(1): 1–27.
Li,B.2026. ModelsTakeNotesatPrefill:KVCacheCanBe
EditableandComposable.arXiv preprint arXiv:2606.17107.
Li, M.; Liu, S.; Fu, D.; Wang, H.; Xia, Y.; Li, H.; Yan, H.;
and Li, P. 2026. KVEraser: Learning to Steer KV Cache
for Efficient Localized Context Erasing.arXiv preprint
arXiv:2606.17034.
Lu, S.; Wang, H.; Rong, Y.; Chen, Z.; and Tang, Y. 2025.
Turborag:Acceleratingretrieval-augmentedgenerationwith
precomputed kv caches for chunked text. InProceedings
of the 2025 Conference on Empirical Methods in Natural
Language Processing, 6599–6612.
Ma, B.; Eitzinger, J.; and Koestler, H. 2026. Leyline: KV
Cache Directives for Agentic Inference. arXiv:2606.01065.
Ouyang,J.;Pan,T.;Cheng,M.;Yan,R.;Luo,Y.;Lin,J.;and
Liu, Q. 2025. Hoh: A dynamic benchmark for evaluating
the impact of outdated information on retrieval-augmented
generation.InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long
Papers), 6036–6063.
Packer,C.;Wooders,S.;Lin,K.;Fang,V.;Patil,S.G.;Stoica,
I.; and Gonzalez, J. E. 2023. MemGPT: Towards LLMs as
operating systems.arXiv preprint arXiv:2310.08560.
Pan, Z.; PATEL, A. D.; Shen, Y.; Hu, Z.; Guan, Y.; Li, W.-
L.;Qin,L.;Wang,Y.;andDing,Y.2026. KVFlow:Efficient
prefixcachingforacceleratingLLM-basedmulti-agentwork-
flows.Advances in Neural Information Processing Systems,
38: 126246–126265.
Wang, S.; Chen, J.; Pan, Y.; Huang, H.; Hao, Y.; Zou, X.;
Xia,W.;Zhang,W.;Wang,H.;Li,J.;etal.2026. ProphetKV:
User-Query-Driven Selective Recomputation for Efficient
KVCacheReuseinRetrieval-AugmentedGeneration.arXiv
preprint arXiv:2602.02579.
Yang,H.;Zhang,R.;Huang,M.;Wang,W.;Tang,Y.;Li,Y.;
Liu,Y.;andZhang,D.2025. Kvshare:Anllmservicesystem
withefficientandeffectivemulti-tenantkvcachereuse.arXiv
preprint arXiv:2503.16525.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A
dataset for diverse, explainable multi-hop question answer-
ing. InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, 2369–2380.
Yao,J.;Li,H.;Liu,Y.;Ray,S.;Cheng,Y.;Zhang,Q.;Du,K.;
Lu, S.; and Jiang, J. 2025. Cacheblend: Fast large language
modelservingforragwithcachedknowledgefusion. InPro-
ceedings of the twentieth European conference on computer
systems, 94–109.
Ye, H.; Gao, Z.; Ma, M.; Wang, Q.; Fu, Y.; Chung, M.-Y.;
Lin, Y.; Liu, Z.; Zhang, J.; Zhuo, D.; et al. 2026. Kvcomm:
Online cross-context kv-cache communication for efficient
llm-based multi-agent systems.Advances in Neural Infor-
mation Processing Systems, 38: 17882–17928.