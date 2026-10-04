# Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment

**Authors**: Josepha Michiko Leo, Hyun-seok Min, Yehoon Jang, Irvan Zidny, Jin-Woo Chung, Sungchul Choi

**Published**: 2026-09-29 02:37:33

**PDF URL**: [https://arxiv.org/pdf/2609.36550v1](https://arxiv.org/pdf/2609.36550v1)

## Abstract
Retrieval-augmented generation is widely used in professional writing, yet whether retrieval grounds revision or merely injects templates is rarely tested where "correct" has a definable meaning. Patent claim amendment supplies that signal: the examiner names the attacked limitation and cites prior art, providing per-case ground truth. We release three artifacts: (i) a corpus of 7,385 USPTO prosecution cases with XML-aligned pre/post claims, rejection, and cited prior art; (ii) a seven-probe battery comparing random and structural-match retrieval as two policies under a fixed prompt scaffold; (iii) a deterministic five-channel metric (C1-C3 and C5 in main, C4 supplementary) requiring no LLM evaluation. Across 9,600 pre-registered calls on four frontier LLMs (Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4, GPT-4o-mini), no tested model exhibits detectable classical prior-injection behavior; retrieval effects are small and direction-inconsistent between random and structural retrieval, and the null is unchanged under a dense (semantic) retriever, across retrieval depths k in {1,3,5,10}, and under a paraphrase-sensitive grounding metric. Revision locality reveals a model-specific difference that the template channel misses. The four-cell taxonomy, which we treat as exploratory, leaves the prior-injector cell unoccupied.

## Full Text


<!-- PDF content starts -->

Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented
Patent Claim Amendment
Josepha Michiko Leo1*, Hyun-seok Min2*, Yehoon Jang1,
Irvan Zidny1,Jin-Woo Chung3,Sungchul Choi1,4†
1Major in Industrial Data Science & Engineering,
Department of Industrial and Data Engineering, Pukyong National University
2Tomocube Inc.3Connectionary4Teamreboott Inc.
{jmichikoleo, jangyh0420, zidny4399}@pukyong.ac.kr
sc82.choi@pknu.ac.kr,min6284@gmail.com,jwchung@connectionary.io
Abstract
Retrieval-augmented generation is widely used
in professional writing, yet whether retrieval
grounds revision or merely injects templates is
rarely tested where “correct” has a definable
meaning. Patent claim amendment supplies
that signal: the examiner names the attacked
limitation and cites prior art, providing per-case
ground truth. We release three artifacts: (i) a
corpus of 7,385 USPTO prosecution cases with
XML-aligned pre/post claims, rejection, and
cited prior art; (ii) a seven-probe battery com-
paring random and structural-match retrieval as
two policies under a fixed prompt scaffold; (iii)
a deterministic five-channel metric (C1–C3 and
C5 in main, C4 supplementary) requiring no
LLM evaluation. Across 9,600 pre-registered
calls on four frontier LLMs (Claude Sonnet 4,
Claude Haiku 4.5, GPT-5.4, GPT-4o-mini), no
tested model exhibitsdetectableclassical prior-
injection behavior; retrieval effects are small
and direction-inconsistent between random and
structural retrieval, and the null is unchanged
under a dense (semantic) retriever, across re-
trieval depths k∈ {1,3,5,10} , and under a
paraphrase-sensitive grounding metric. Revi-
sion locality reveals a model-specific difference
that the template channel misses. The four-cell
taxonomy, which we treat as exploratory, leaves
the prior-injector cell unoccupied.
1 Introduction
Automatic generation of structured technical docu-
ments is expanding faster than its evaluation infras-
tructure. Recent end-to-end systems generate hy-
potheses (Gottweis et al., 2026), write manuscripts
(Song et al., 2026), and conduct their own peer re-
view (Lu et al., 2026): Lu et al. (2026) report a gen-
erated manuscript passing workshop peer review,
and more than 120 computer-generated papers were
retracted from IEEE and Springer journals over a
*Equal contribution.
†Corresponding author.decade ago (Van Noorden, 2014). By comparison,
code generation matured only after HumanEval
(Chen et al., 2021) and SWE-bench (Jimenez et al.,
2024) introduced structurally rigorous evaluation
tied to executable ground truth.
Patents provide that anchor: examiners apply
shared legal standards (35 U.S.C. §102 novelty,
§103 non-obviousness, §112 enablement) under
the standardized Manual of Patent Examining Pro-
cedure (MPEP), creating an experimentally con-
trolled setting in which correctness is legally defin-
able. The USPTO receives over 612,000 new utility,
plant, and reissue patent applications annually and
processes more than 137,000 Requests for Contin-
ued Examination (U.S. Patent and Trademark Of-
fice, 2025). Patent prosecution is a paradigm case
of grounded revision: each amendment is paired
with an examiner-identified attacked limitation and
the on-record passage of a cited reference, an align-
ment most revision corpora lack.
Commercial systems such as PatentGPT (Pat-
snap) and IPRally Drafter, along with a growing
set of startups, market retrieval-augmented amend-
ment generation on the premise that retrieving prior
examples grounds the output; to our knowledge
these are vendor product claims advanced without
peer-reviewed evaluation, which is part of what
motivates an independent test. Whether retrieval
substantively grounds outputs in structural simi-
larity, or serves more as an undifferentiated prior-
injection anchor whose effect is independent of
the specific items retrieved, is an open empirical
question.
Academic work has progressed in parallel but
has not tested whether retrieval substantively en-
gages with the rejection or merely produces surface-
plausible prose. Patent-CR (Jiang et al., 2025a)
releases 22,606 draft-to-grant pairs but evaluates
only with surface-level similarity. PANORAMA
(Lim et al., 2025) enumerates claim revision with-
out evaluation. PEDANTIC (Knappich et al., 2025)
arXiv:2609.36550v1  [cs.CL]  29 Sep 2026

1INPUT (4- TUPLE)R e vision set t ing
from USP T OP re - claim
(Original claim)R eject ion
(Offic e act ion)P rior ar t
(Cita t ions)Gold p os t -amendment
(R ef erenc e)4- T uple C orpus
7,385 USP T O c ases
XML -alingne d2PR OBE B A T TER Y (7 PR OBES )Same re vision task ,
diff erent retrie v al s tr uctureA —EO b ser v at ion pro b esABCDE
Cita t ion
GroundingSc op e
P reser v a t ionStr uctural
C onsis tencyF ,  GR etrie v al p er tu b at ionFGStr uctural  R etrie v alRandom  R etrie v al3CHANNEL METRICSQuant if y b eha vioral shifts
(ΔF v s ΔG)P er - AmendmentC 1Groundin g
A lignment-0.0057-0.0059C 2R e vision
L o c alit y-0.0044-0.0076C 3Sc op e
P reser v a t ionT empla te
D ep endenc eC 5~0.20~-0.27* gpt -4 o - miniΔF  (Random)ΔG  (Str uctural)P er - P ro b e - P airC 4R o b us tnes s4T A X ONOMYClas sif y b y ΔF and ΔG+Anch o r
G rou nd e r

ΔF<0 , ΔG<0St r u ct ur al
Match e r

ΔF>0.2 , ΔG>0.2ΔG  (Str uctural  R etrie v al)Mi l d  P r i or 
Inj e ct i o n

0.2  ≥  ΔF , ΔG  ≥  0P r i o r
Inj e ct i o n

ΔF>0.2 , ΔG>0.2−−ΔF  (Random  R etrie v al)+Opp osite signs 
➔ s tr ucture -sensit iv e ,
not -template dR eject ion
R esolut ionE dit
Minimalit y0.0210.0210.0280.028
o v erall  h it - ra te0.56Figure 1: Does the model address the examiner-attacked limitation (top, grounded revision) or recycle generic
patent phrasing (bottom, prior injection)? Our probe battery (§4.1) and metric (§4.2) distinguish the two without an
LLM judge.
and PILOT-Bench (Jang et al., 2025) classify §112
indefiniteness and rejection-issue types respec-
tively, without generation. None tests retrieval’s
role in grounded revision under a controlled policy
contrast. Plausibility is cheap: any competent gen-
erator produces patent-sounding text, but grounded
revision additionally requires that the edit target
the attacked limitation, differentiate from the cited
mechanism, and neither over-narrow nor recycle
boilerplate. These failure modes leave textual fin-
gerprints (MPEP §714), and we exploit them to
build a probe-based evaluation protocol.
Contributions.The paper contributes three cou-
pled components.
Corpus (§3).We release 7,385 four-tuples (pre-
amendment claim, rejection, cited prior art, post-
amendment claim) aligned at the XML level from
the USPTO Open Data Portal, supporting a strati-
fied 100-case test cohort (§5.1) and sampling-bias
robustness checks (Appendix K).
Probe battery (§4.1).We design seven
probes: five observational (A–E) and two retrieval-
intervention (F, G) that realize a random-versus-
structural policy contrast under a fixed prompt scaf-
fold (Figure 1). The observational axis makes the
F-versus-G comparison interpretable rather than
merely measurable.
Metric (§4.2).We define a five-channel deter-
ministic metric with no LLM in the evaluation loop.C1–C3 and C5 are per-amendment channels and
carry the main-paper verdicts; C4 (robustness) is
defined on probe pairs and is reported in supple-
mentary tables (§4.2).
Together these test three pre-registered hypothe-
ses (H1–H3, §5.3) in a 2 ×2 factorial across Claude
and OpenAI families at flagship and smaller tiers.
Scope. We study applicant-side amendment gen-
eration, i.e., given a rejection, generate the post-
amendment claim.
Preview of findings.Across the four tested
LLMs, the prior-injector cell of our retrieval-
mechanism taxonomy turned out to be empty: ran-
dom and structural retrieval produce small, of-
ten opposite-sign shifts in template dependence
rather than the uniform inflation that the most pes-
simistic grounding-as-template-recycling account
would predict. The framework’s value does not rest
on that null: it registers model-specific differences
on revision locality (most clearly for GPT-5.4) that
a single-channel test would miss, and, treating the
four-cell taxonomy as exploratory, locates each
tested model within it.
2 Related Work
2.1 Patent NLP
The closest prior work is Patent-CR (Jiang
et al., 2025a), which released 22,606 rejected-to-
granted claim pairs evaluated with BLEU, ROUGE,

BERTScore, G-Eval, and a five-axis expert rating.
The evaluation scores pair fidelity but does not ex-
pose the examiner’s rejection or the cited prior art,
nor does it contrast retrieval policies under a con-
trolled scaffold. We add aligned rejection and prior-
art channels to the input, and switch from fidelity
scoring to a probe-based policy-contrast evaluation.
Other patent NLP work sits outside a controlled re-
trieval contrast for structural reasons. BIGPATENT
(Sharma et al., 2019) frames patent text as an ab-
stractive summarization corpus; PATENTWRITER
(Shomee et al., 2025) and AutoPatent (Wang et al.,
2024) generate claims or full specifications from
non-conditional inputs; and generation quality is as-
sessed by evaluation-focused work, including Lee
(2023), Jiang et al. (2025b), and PatentScore (Yoo
et al., 2025). On the classification side, PEDAN-
TIC (Knappich et al., 2025) labels §112 definite-
ness, PANORAMA (Lim et al., 2025) enumerates
claim revisions without evaluation, PILOT-Bench
(Jang et al., 2025) classifies rejection-issue types,
and Shi et al. (2025) predicts examination out-
comes. In none of these does the retrieval channel
enter the input aligned with the rejection that re-
trieved exemplars would need to address, so none
admits the policy-contrast test we run. Related re-
sources also align examiner rejections with claims:
ClaimBrush (Kawano et al., 2024) pairs pre/post
claims with Office-Action metadata for Japanese
filings, and Tree-of-Claims (Yu et al., 2025) and
the USPTO Office Action Research Dataset (U.S.
Patent and Trademark Office, Office of the Chief
Economist, 2017) align cited prior art with rejected
claims. Our contribution is more delimited: we
align the pre- and post-amendment claim text, the
rejection, and the cited prior art as four-tuples at
the XML level, with an application-number rebuild
index, and frame them as model input, which is
what allows Probes F and G to be interpreted as a
contrast between two retrieval policies inserted into
the same prompt scaffold. InstructPatentGPT (Lee,
2024) similarly uses Office Actions as a training
signal rather than as a controlled retrieval contrast.
2.2 RAG evaluation and behavioral probing
RAG evaluation has progressed from retrieval-
augmented pre-training (Guu et al., 2020; Lewis
et al., 2020) through few-shot retrieval augmen-
tation (Izacard et al., 2023) and self-reflective re-
trieval (Asai et al., 2024), with automated eval-
uation metrics (Es et al., 2024) and heteroge-
neous zero-shot retrieval benchmarks (Thakur et al.,2021), but remains predominantly correlational.
We borrow intervention logic from behavioral prob-
ing (Elazar et al., 2021; Ribeiro et al., 2020; Vig
et al., 2020): our observational probes (input per-
turbations, detailed as A–E in §4.1) follow Check-
List, and our two retrieval interventions (random
versus structurally-matched retrieval, F and G in
§4.1) are policy-contrast interventions on the re-
trieval channel, in the design tradition of Vig et al.
(2020) but at the level of a comparison between
two retrieval policies inserted into a fixed prompt
scaffold rather than full causal identification of re-
trieval per se. Work on retrieval noise shows that
even random or irrelevant passages can shift gen-
eration substantially (Cuconasu et al., 2024; Fang
et al., 2024), and that context position matters (Liu
et al., 2024); correlational RAG benchmarks, how-
ever, cannot separate similarity-driven grounding
from generic context injection, which our F-versus-
G contrast isolates. The patent-amendment setting
addresses this confound at the corpus level: every
test case carries an examiner-identified attacked
limitation and an on-record cited reference, so the
input is fully specified before retrieval is added,
and any shift produced by F or G is attributable to
the retrieval-policy intervention. This enables the
framework to place models in the four-cell taxon-
omy of §6.3 rather than only ranking them on a
single retrieval-quality scalar.
3 Patent Amendment 4-Tuple Corpus
We release a corpus of 7,385 prosecution cases
in which every amendment is aligned at the XML
level with its triggering rejection, the cited prior
art, and the pre- and post-amendment claim text.
The scale supports the 100-case test cohort (§5.1),
a 4,221-case retrieval pool for Probes F and G, and
a sampling-bias replication reserve (Appendix K).
3.1 Source and alignment
We start from the PILOT-Bench PTAB-appeal sub-
set (Jang et al., 2025) of 13,749 cases and en-
rich each case via the USPTO Open Data Portal
(ODP). For each case we resolve its application
number, retrieve the file-wrapper document list,
and identify the first non-final rejection together
with the immediately preceding Claims filing (pre-
amendment state) and the first Claims filing after
the rejection (post-amendment state). For each
pre/post claim pair we compute a per-claim differ-
ence status (kept, modified, new, cancelled) using

claim-number alignment, and for modified claims
a SequenceMatcher-based similarity ratio plus the
list of added and removed spans. The difference
is the ground truth against which the generated
amendments are scored (§4.2).
3.2 Summary statistics
Two corpus-level findings support the paper’s fram-
ing. First, amendment is predominantly surgi-
cal: 64.5% of per-claim actions are modifications
at median similarity 0.933 ( ∼7% character-level
edit), with cancellations and new claims account-
ing for another 32.5%. Second, the rejection-to-
amendment linkage holds at the parse level: 94.5%
of scorable cases show overlap between rejected
and post-amendment-modified claim numbers. Of
the 7,385 parseable triples, 5,755 admit a case-level
C2 denominator (§4.2). The first-rejection statute
breakdown is dominated by §102 (2,337 cases),
with §112 (1,245), §101 (950), and §103 (738) ac-
counting for the remainder. Full corpus statistics
are reported in Appendix A.4.
3.3 Release
We release the parsed JSONL corpus and pars-
ing code under a permissive open license; an
application-number index permits zero-cost recon-
struction from USPTO ODP, whose public data
have no copyright restriction. Our corpus is the
only one of Patent-CR, PANORAMA, PEDANTIC,
and PILOT-Bench that aligns all four elements (re-
jection context, cited prior art, pre/post claim pair,
and amendment diff) at the XML level. Patent-CR
provides the pre/post pair but omits the rejection
and prior-art channels; PANORAMA, PEDANTIC,
and PILOT-Bench target judgment, classification,
or upstream retrieval rather than amendment gener-
ation.
4 Method
Our method has two components. The probe bat-
tery (§4.1) defines seven controlled input manip-
ulations: five observational baselines (A–E) plus
a two-policy retrieval contrast (F vs G) inserted
into a fixed prompt scaffold, with the F/G compari-
son interpreted against the observational baseline
rather than as unconditional causal identification
of retrieval. The five-channel deterministic metric
(§4.2) scores every probe condition on common
axes without using an LLM as judge.4.1 Probe Battery
We design two probe families read jointly. Ob-
servational probes A–E characterize each LLM’s
baseline amendment signature under input pertur-
bations. Retrieval probes F and G intervene on the
retrieval channel via a controlled random-versus-
structural policy contrast under a fixed insertion
template. The observational axis supplies the in-
terpretive frame against which the retrieval-policy
effect is measured, so an F-versus-G shift is read
relative to a characterized baseline rather than an
undifferentiated mean. All probes share a common
input template (pre-amendment claim, rejection ra-
tionale, cited prior art) and differ only in what is
manipulated; each probe’s expected effect on the
five-channel score (§4.2) is fixed in advance.
Observational probes (A–E). Probe A (Claim
Truncation)modifies a single decisive limitation
in the rejected independent claim by deletion, ad-
dition, paraphrase, or antonym swap. A grounded
model follows the perturbation in C1; a model that
ignores the rejection signal will show C1 invari-
ance.
Probe B (Rationale Shuffling)compares the
original examiner findings against a version with
one sentence replaced by a semantically inconsis-
tent fragment or with sentence order permuted. A
rationale-grounded model holds C1 stable across
the coherence break, while a surface tracker de-
grades.
Probe C (Decoy Citation)substitutes either a
wording-similar reference that does not teach the
mechanism, or a wording-dissimilar reference that
does, separating lexical from mechanistic engage-
ment with the cited art.
Probe D (Boilerplate Injection)prepends
canonical patent boilerplate to the drafting context.
A grounded model treats the prepended phrasing
as scaffolding and leaves C5 and C3 largely un-
changed; a prior-injection-prone model inflates C5.
Probe E (Drafting Hint)adds one of four task
hints: “make minimal amendment”, “preserve
scope”, “focus only on novelty”, or “avoid unneces-
sary narrowing”. A model with stable instruction-
following should produce measurable shifts on C2
and C3 in the directions the hints predict.
Retrieval probes (F, G). Probe F (Random Re-
trieval)retrieves kpast amendments from the cor-
pus pool (excluding the test case and its ancestors),
selected uniformly at random, and prepends them

as context. Random retrieval realizes the prior-
injection condition in its strongest form: any effect
cannot be attributed to similarity-driven grounding
because selection is similarity-agnostic by construc-
tion.
Probe G (Structural-Match Retrieval)re-
trieves kpast amendments matched on (i) statute
section, (ii) statute subsection, and (iii) a coarse
limitation-pattern feature derived from the at-
tacked claim. The deterministic match isolates the
structural-similarity signal: any shift attributable
to the match itself must surface in the F-versus-G
delta.
Both probes use the same k(default 3; retrieval-
depth sensitivity in Appendix R) and the same
prompt-insertion template; G uses a determinis-
tic feature-based matcher rather than a dense re-
triever to keep the F and G contrast interpretable as
a structural-similarity test (Appendix F). Because
F, G, and the no-retrieval baseline all sit inside the
same prompt scaffold, F-versus-G identifies the ef-
fect of switching retrieval policy under a fixed scaf-
fold rather than the unconditional effect of adding
retrieval. The empty prior-injector cell that the
framework finds in §6.3 is therefore a finding about
the deterministic structural-similarity policy real-
ized by Probe G, not about retrieval-augmented
generation in general; dense-retriever configura-
tions would constitute a different point in the same
policy-contrast design and can be evaluated with
the released framework without modification.
4.2 Five-Channel Evaluation Metric
Using an LLM as evaluator embeds the evaluator’s
biases into the score, a concern made empirical by
our findings (§6): the four frontier models clus-
ter into distinct behavioral patterns under retrieval
perturbation, so selecting any one as judge would
install that model’s signature as the evaluation axis.
Therefore, we adopt a deterministic five-channel
metric. Each channel is a function of text, parser
output, and the XML gold post-amendment claim;
channels are designed to be jointly informative,
and we verify their near-independence empirically
(§6.4).
Channels. C1 (grounding alignment)measures
the overlap between limitations modified in the
generated amendment and limitations named as
rejected in the office action, computed at limitation-
token level after claim-number alignment. A high
C1 means the model is editing the same limitationthe examiner attacked.
C2 (revision locality)is the ratio
editdist(gen,pre)/editdist(gold_post,pre) ;
a value of 1 indicates the model edited at the same
scale as the gold amendment, ≪1 indicates under-
editing, and ≫1 indicates over-rewriting. C2
catches timidity and over-rewriting as numerically
distinct rather than collapsing them into a single
similarity score.
C3 (scope preservation)is the Jaccard over-
lap of noun phrases between the generated amend-
ment and the original claim’s invention core, after
limitation-removal normalization; reduction indi-
cates over-narrowing. The normalization isolates
true scope drift from the mechanical narrowing that
any limitation deletion induces.
C4 (robustness)measures, for any pair of probe
conditions, whether the amendment shifts in the
predicted direction, scored per probe and averaged
within a model. C4 is defined on probe pairs rather
than on individual amendments, so we report it in
supplementary material (per-pair tables released
alongside Appendix Table 4) to keep the main-
paper tables dimensionally consistent with the per-
amendment channels C1–C3 and C5. The headline
verdicts in §6.1 and the taxonomy in §6.3 are there-
fore computed on the per-amendment channels; C4
enters only as a supplementary directional-hit-rate
check. On the 100-case cohort, the grand C4 across
the seven pre-registered (probe, channel) pairs is
0.56 (n=2,276 case-model contrasts; per-pair table
in Appendix G).
C5 (template dependence)is the frequency of
canonical amendment phrases mined from the re-
trieval pool, normalized per 1,000 characters of
generated claim text. C5 is the central channel for
adjudicating the prior-injection hypothesis: if re-
trieval inflates the rate of canonical phrases, C5
rises in lockstep. We emphasize that C5’s absolute
level is not a measure of prior injection: canoni-
cal phrasing is normal in competent drafting, so a
high C5 may reflect legitimate convention. Prior
injection is diagnosed only from ∆C5, the retrieval-
induced change over the no-retrieval baseline for
the same case and model, so convention present
at baseline is differenced out and only retrieval-
added phrasing is attributed to injection. C5 re-
mains a proxy that cannot alone separate appropri-
ate phrasing from recycling, and a lower C5 need
not indicate better grounding; criterion validation
is deferred to the post-review study (§4.2).

Validity.The metric design follows the multitrait-
multimethod logic of Campbell and Fiske (1959):
each channel ties to a specific XML-record feature
rather than an LLM judgment. Three legs support
the metric: (i) within-case construct checks where
each channel moves in the direction a patent practi-
tioner would predict (Appendix C); (ii) empirical
near-independence pooled across 3,062 (case, con-
dition, model) triples, all off-diagonal correlations
|ρ| ≤0.29 , below the pre-registered |ρ|>0.7
flag threshold (Table 3); (iii) a 30–50-case double-
rated attorney study committed post-review (a 10-
case pre-release sanity sample showed all four
channels moving in the direction the reading attor-
ney predicted; Appendix J). The construct-check
and channel-independence legs carry the metric-
validity argument in this paper; the formal κleg is
deferred to the post-review study.
5 Experiments
We fix the entire experimental protocol before any
model call is made, following the pre-registration
discipline recommended for probe-based evalua-
tions (van Miltenburg et al., 2021).
5.1 Cohort and Retrieval Pool
The 100-case test cohort is constructed via joint
stratified sampling across six axes (subdecision out-
come, rejection statute, technology center, amend-
ment pattern, XML format era, decision year bin)
matching the marginal distributions of the 4,270-
case eligibility pool while guaranteeing minimum
cell sizes (Appendix N). The remaining 7,285
cases, after filtering for resolvable attacked-claim
alignment, yield a 4,221-case retrieval pool indexed
by (statute section, statute subsection, limitation
pattern), used by Probes F and G. The cohort and
pool are non-overlapping by construction.
5.2 Model matrix
We run a 2 ×2 factorial across vendor family and
model tier. Stage 1 (Claude Sonnet 4 and GPT-
5.4 flagships) runs the full probe battery. Stage
2 extends to Claude Haiku 4.5 and GPT-4o-mini,
conditional on Stage 1 effect-size criteria detailed
in Appendix K.1. Each case is scored under 8 con-
ditions ×3 replicates (24 evaluations per case per
model), giving within-model paired comparisons
with>80% power at d= 0.3 ,α= 0.05 . Infer-
ence settings: temperature 0.3 for all models; three
independent replicates per (case, probe, model)
with separate sampling seeds; retrieval k= 3(retrieval-depth sensitivity over k∈ {1,3,5,10} in
Appendix R).
5.3 Pre-registered hypotheses
H1 (retrieval as prior-injection anchor): Probe F
(Random Retrieval) increases template dependence
(C5) relative to baseline by at least 0.2 and does not
improve grounding alignment (C1) by more than
0.1. This is the strongest commercial concern about
retrieval-augmented drafting in operational form:
if random exemplars inflate template phrasing with-
out sharpening grounding, retrieval is acting as a
vendor-agnostic boilerplate anchor.
H2 (similarity as second-order): Probes F (ran-
dom) and G (structural) yield statistically indistin-
guishable shifts in C1, C2, and C5 ( |∆F−∆G| ≤
0.1on each channel, or same sign with no signif-
icant ranking). H2 isolates the marginal value of
structural matching: if F and G are observationally
equivalent, the structural-similarity layer adds noth-
ing the random baseline does not already provide.
H3 (template-recycling inflation): retrieval does
not reduce template dependence below baseline on
either probe; equivalently, the grounded-revision
hypothesis that retrieval reduces generic phrasing
is not supported. H3 closes the loop in the opposite
direction from H1, ruling out the optimistic “re-
trieval makes models write more like experts” story
when both probes leave C5 at or above baseline.
The full 3×3 outcome space (each hypothesis ∈
{supported, rejected}) is addressed in §6.3.
6 Results and Analysis
We ran the benchmark on four frontier LLMs
(Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4,
GPT-4o-mini) across 100 cohort cases ×8 con-
ditions ×3 replicates = 9,600 model calls. Parse-
success rate was 97.4% overall (93.0% on GPT-4o-
mini to 100% on GPT-5.4). Per-response scores are
aggregated as per-case mean across replicates, then
median across cases within (model, condition).
6.1 Pre-registered hypothesis verdicts
No model shows detectable prior injection.H1
(retrieval-as-prior-injection) is not supported across
all four models: ∆C5 falls below the pre-registered
0.2 threshold in every case (max +0.20 on GPT-
4o-mini, at the threshold). Under the pre-registered
convention the threshold is the supported-side
boundary, so ∆C5 = +0.20 is marked “ ∼” rather
than “Y”. Paired Wilcoxon signed-rank tests with

Table 1: Pre-registered hypothesis verdicts on C5
(canonical-phrase rate per 1,000 characters, N=300 per
cell = 100 cases ×3 replicates). Y: supported; N: not
supported; ∼: at-boundary, see §6.1 for the boundary-
handling rule.
Model base∆F∆G H1 H2 H3
Claude Sonnet 4 3.99−0.16−0.13NYN
Claude Haiku 4.5 3.40+0.19 +0.03N NY
GPT-5.4 3.73−0.12 +0.06N N N
GPT-4o-mini 3.14+0.20−0.27∼N N
Holm–Bonferroni correction and bootstrap 95%
CIs on the headline ∆C5 values (Appendix O) find
no significant shift: 0 of 8 baseline-versus-retrieval
comparisons (F and G across four models) reach
uncorrected significance, and none survives correc-
tion. A two-one-sided-tests (TOST) equivalence
check against the pre-registered ±0.2 margin fur-
ther establishes that the median ∆C5 is statistically
equivalent to zero for all four models under both
probes, so the GPT-4o-mini boundary is a tested
equivalence rather than only an adjacency to thresh-
old. We emphasize that equivalence within the
pre-registered ±0.2 margin bounds the effect but
does not establish its absence, and in particular
does not exclude a true shift lying just below +0.2 ;
we therefore report nodetectableprior injection at
this sample size and margin rather than its absence.
Because any fixed cutoff is arbitrary, the verdict
does not rest on a point estimate crossing 0.2: no
∆C5 is significant or distinguishable from zero (Ap-
pendix O), so no model shows asignificantincrease
at any threshold in 0.15–0.25, though the Haiku 4.5
and GPT-4o-mini point estimates sit near 0.2. H2
is uniquely supported by Sonnet 4: both F and G
produce same-sign shifts within 0.1 (both modestly
negative). The other three models show opposite-
sign effects (GPT-5.4, GPT-4o-mini) or disparate
magnitudes (Haiku 4.5); the pre-registered 0.10
bound is a strict bound, but H2 is jointly defined
with the no-significant-ranking clause, so Haiku’s
0.16 gap fails the joint criterion ordinally as well
and is recorded as “N” rather than at-boundary “ ∼”.
H3 holds only for Haiku 4.5; Sonnet 4 and GPT-
4o-mini each reduce template dependence under
at least one retrieval condition. Per-case effects
on C5 reach ±2to8even where aggregate me-
dians stay sub-threshold; two worked examples
are in Appendix C.6. GPT-4o-mini over-edits at
baseline (C2=1.46), and smaller-tier models show
stronger Probe A grounding shifts than flagships
(full probe×channel deltas in Appendix Table 4).6.2 Model divergence on revision locality
Although H1 is null on the template channel, the
benchmark registers model-specific differences on
revision locality (C2). Table 2 shows retrieval re-
ducing edit magnitude in both Claude models and
in GPT-4o-mini, while GPT-5.4 slightly increases
it, so the pattern is GPT-5.4 as the lone outlier
rather than a clean Claude-versus-GPT split. We
therefore treat this as a model-specific, exploratory
observation rather than a vendor-level finding. Un-
der paired Wilcoxon tests, 3 of 8 C2 comparisons
are nominally significant (Haiku 4.5 F and G, Son-
net 4 F), all decreases, and none survives Holm
correction (Appendix O), so the effect is carried by
Haiku 4.5. The point that a shift invisible on C5
surfaces on C2 still holds, showing the battery is
not simply insensitive; we do not, however, claim
a vendor-aligned family split.
Table 2: Revision-locality (C2) effects under random
(F) and structural (G) retrieval, per model.
Model base C2∆F∆G
Claude Sonnet 4 0.617−0.074−0.013
Claude Haiku 4.5 1.147−0.174−0.189
GPT-5.4 0.789+0.016 +0.058
GPT-4o-mini 1.459−0.044−0.076
6.3 Where the evidence places retrieval
The pre-registered 2×2×3 framework places
four retrieval-mechanism accounts in play. The
main experiment resolves them as follows, with
each cell tagged by its (∆F,∆G)coordinate on
C5 from Table 1.(a) Anchor-grounder(H1 re-
jected, H3 rejected, retrieval reduces templateness
without inflating alignment): Claude Sonnet 4 at
(−0.16,−0.13) .(b) Prior injector(both ∆Fand
∆Gabove +0.2 ): not observed at pre-registered
effect size.(c) Structural matcher( F̸=G , H2
rejected): GPT-5.4 at (−0.12,+0.06) (sign flip)
and GPT-4o-mini at (+0.20,−0.27) (0.47 direc-
tional gap).(d) Inert / mild prior injector(H1
below threshold, H2 mild): Claude Haiku 4.5 at
(+0.19,+0.03) (Figure 2). The “ ∼” verdict for
GPT-4o-mini on H1 and its cell-(c) assignment
are complementary projections of the same point:
cell-(b) membership requires the joint (∆F,∆G)
pair to clear threshold with consistent sign. This
joint criterion is a post-hoc interpretive overlay,
specified at the taxonomy step once the headline
cell turned out empty, rather than a separately pre-
registered threshold. Haiku 4.5’s cell-(d) tag fol-
lows the same overlay: H1 fails strictly and H2 fails

with|∆F−∆G|= 0.16 (above 0.1, same sign),
so the F-versus-G pair is mild-but-not-equivalent
rather than the same-sign-within-0.1 signature of
cell (a).
Anchoring F and G against the Probe D boil-
erplate baseline.The cell assignments acquire
interpretive weight when read against the observa-
tional deltas of Probes A–E for the same model
(Appendix Table 4). Probe D (Boilerplate Injec-
tion) provides the closest in-corpus reference for
a generic prepended-context effect: on Sonnet 4 it
moves ∆C5 by+0.03 , on Haiku 4.5 by −0.09 , on
GPT-5.4 by +0.07 , and on GPT-4o-mini by −0.07 .
For three of four models (Sonnet 4, GPT-5.4, GPT-
4o-mini) the post-hoc descriptive gap |∆F|−|∆ D|
is bounded at 0.13; we use absolute values here
because ∆Fand∆Ddisagree in sign on these
models, so the comparison is one of prepended-
context dosage rather than signed direction. At this
dosage, random retrieval anchors C1/C2 toward
the boilerplate prior rather than behaving qualita-
tively differently. Haiku 4.5 is the per-model excep-
tion, with ∆F= +0.19 exceeding |∆D|= 0.09
(cell-(d) assignment; full per-model breakdown in
Appendix E). Probe C (Decoy Citation) similarly
bounds the scale of mechanism-similarity confu-
sion that any retrieval channel could induce, pro-
viding the C1 reference against which the near-zero
∆C1 values under F and G are read as null rather
than as insensitivity.
Distinguishing anchor-grounder from inert.
Sonnet 4 (cell a) and Haiku 4.5 (cell d) are sep-
arated by two signals: the F-versus-G C5 sign
pattern and C2. Sonnet 4 has same-sign ∆F,
∆Gwithin 0.1 (H2-supported), with small same-
signed C2 shifts. Haiku 4.5 has ∆F= +0.19 ,
∆G= +0.03 (H2 rejected), with the largest abso-
lute∆C2in the matrix (−0.174,−0.189).
Small, direction-inconsistent effects.The
modal retrieval behavior is therefore (c) or (d):
small in magnitude and often direction-inconsistent
between F and G. Discrimination on revision
locality (C2; §6.2) confirms that the framework
detects retrieval-policy effects where they occur
rather than being simply insensitive. Sonnet 4’s
H2 uniqueness ( F≈G , both slightly reducing
template phrases) is the behavioral pattern
most aligned with the originally conjectured
grounded-revision mode.
GPT-5.4
(-0.12, +0.06)
0 . 0
0 . 2
- 0 . 2ΔG on C5 (Structural Retrieval)
Prior Injector 
Both Δ  > 0.2 
UNOCCUPIED
GPT-4o-mini
(+0.2,-0.27)
- 0 . 2
0 . 0
0 . 2Claude Sonnet 4
(-0.16, -0.13)
Claude Haiku 4.5
(+0.19, +0.03)
ΔF on C5 (Random Retrieval)Structural Matcher
(Opposite signs Δ F vs Δ G) 
Anchor-Grounder
(both Δ F, Δ G <0)Structural Matcher
(Opposite signs Δ F vs Δ G) 
Mild prior injector
(both 0.2 ≥ Δ F, Δ G ≥ 0 ) Figure 2: Four retrieval-mechanism cells in (∆F,∆G)
space on C5. Coordinates from Table 1: Son-
net 4 (−0.16,−0.13) anchor-grounder; GPT-5.4
(−0.12,+0.06) and GPT-4o-mini (+0.20,−0.27)
structural-matcher (opposite-sign F vs G); Haiku 4.5
(+0.19,+0.03) inert / mildly prior-injecting. The prior-
injector cell (both ∆>+0.2 ) is unoccupied across all
four models tested in this four-model, one-corpus, k=3,
deterministic-structural-matcher configuration.
6.4 Benchmark methodology and metric
validity
Small samples mislead.A smoke run at N=10 per
cell gave ∆C5=+0.84 for GPT-5.4 under F (H1-
supported region); at N=300 the same effect col-
lapses to −0.12 (Appendix D). This sign-reversing
reversal motivates the N=100 -per-condition-with-
replicates sizing used throughout.
Channels are near-independent.Pooled across
3,062 (case, condition, model) triples, between-
channel correlations are low throughout. No pair
exceeds the pre-registered |ρ|>0.7 flag thresh-
old (Table 3). The strongest relationship is C2–
C3 Spearman at −0.29 , consistent with the con-
struct distinction: larger edits mechanically touch
more of the claim and can reduce scope overlap.
The four channels are therefore empirically near-
independent on the main data, supporting the §4.2
design claim that they are jointly informative rather
than redundant.
Directional consistency (C4).The grand C4 of
0.56 (Table 5) splits into two kinds of near-chance
cell. The retrieval-on-template cells (F,C5 0.50;
G,C5 0.47) sit at chancebecausethey measure the
prior-injection null; the manipulation-check cells
(D,C5 0.42; Probe C for C1 at 0.40/0.32) are posi-
tive controls whose below-chance rates are a gen-

Table 3: Channel correlations pooled across 3,062 (case,
condition, model) triples. Pearson above the diagonal,
Spearman below.
C1 C2 C3 C5
C1 —+0.04−0.21−0.05
C2+0.07—−0.14−0.04
C3−0.17−0.29—+0.06
C5−0.01 +0.03 +0.04—
uine tuning limitation, bearing on C1’s sensitiv-
ity, while the FLAT predictions are well-calibrated
(B,C1 0.95; B,C3 0.64). C4 is supplementary and
carries none of the headline verdicts, which rest on
the per-amendment channels C1–C3 and C5.
6.5 Three commercial properties not
confirmed
The empirical pattern in §6.3 does not support
three commonly advertised properties of retrieval-
augmented drafting, against pre-registered testing
(H1 and H3 against fixed effect-size thresholds, H2
against a sign-and-magnitude equivalence condi-
tion).Retrieval grounds the output: on ground-
ing alignment (C1), |∆C1| ≤0.014 under both F
and G across all four models, and no model shows
a significant positive ∆C1 under either probe af-
ter Holm correction; the same null holds under
a paraphrase-sensitive semantic C1 (Appendix P).
Structural retrieval beats random retrieval: H2
is rejected for three of four models; only Sonnet 4
satisfies F≈G .Retrieval makes models write
more like experts: the maximum observed |∆C5|
is 0.20 (boundary on H1’s threshold), and no model
clears a symmetric −0.20 “retrieval reduces tem-
plate” threshold either. These findings do not en-
tail that retrieval-augmented patent drafting fails
in general; they entail that the three properties are
not confirmed in this instrument, and the released
framework can be applied to other (model, corpus,
retriever) configurations.
6.6 Robustness of the null
The prior-injection null is stable along three axes
we varied after the main run.Retrieval mech-
anism.A dense (semantic) realization of Probe
G, embedding-cosine top-3 over the same pool
withk, exemplar format, and base prompt held
fixed, retrieves far more semantically similar exem-
plars than either F or G (mean query–exemplar co-
sine 0.87 vs 0.59–0.61) yet moves nothing: across
1,200 calls (0 failures) on all four models, 0 of 16
baseline-versus-dense comparisons survive Holmcorrection and no model shows a significant C5
increase (Appendix Q). The null therefore spans
three retrieval regimes (random, structural, dense),
not just the deterministic matcher.Retrieval depth.
Varying only k∈ {1,3,5,10} with the cohort, re-
triever, scaffold, exemplar ordering, model, and de-
coding held fixed leaves C1 practically stable and
yields no monotonic increase in C5; applying the
pre-registered criteria at every depth gives the same
qualitative verdicts, so the results are not an artifact
ofk=3 (Appendix R).Grounding metric.The
grounding null holds under a paraphrase-sensitive
semantic C1 that flags different cases than the lexi-
cal version (Spearmanρ=0.37; Appendix P).
7 Conclusion
We presented a probe-based evaluation of rejection-
grounded patent claim amendment: a corpus of
7,385 four-tuples from the USPTO Open Data Por-
tal, seven probes, and a five-channel deterministic
metric (C1–C3 and C5 carry the verdicts; C4 is sup-
plementary). In this four-model, one-corpus setting,
pre-registered testing finds no model above the H1
prior-injection threshold (max ∆C5= +0.20 ), a
null that holds under random, structural, and dense
retrieval and across k∈ {1,3,5,10} (§6.6); H2
holds only for Sonnet 4, H3 only for Haiku 4.5,
and revision locality (C2) reveals a model-specific
difference (clearest for GPT-5.4) the template chan-
nel misses. We release the framework, corpus, and
scoring code so the protocol can be applied to com-
mercial stacks and to other retrieval configurations,
where cell occupancy may differ.
Limitations
Scope and generalization.We examine a single
rejection-amendment round; real prosecution can
span multiple rounds with accumulating examiner
commitments that our probes do not capture. Find-
ings come from one domain (U.S. patent amend-
ment) and four models; although we corroborate
the null across random, structural, and dense re-
trieval and across retrieval depths k∈ {1,3,5,10}
(§6.6), generalization to other structured-document
revision tasks such as legal briefs or scientific revi-
sion is conjectural and reserved for follow-up work.
The retrieval pool is also drawn from the same cor-
pus distribution as the test set, whereas a deployed
commercial retriever would index a different cor-
pus; we therefore read our retrieval effects as an
upper bound on best-case benefit.

The empty prior-injector cell is conditional on
the matcher.The headline finding that the prior-
injector cell of the taxonomy is empty is condi-
tional on Probe G’s deterministic feature-based
matcher (statute section ×subsection ×coarse
limitation-pattern), chosen so that the F-versus-G
contrast is interpretable as a structural-similarity
test rather than a retriever-quality test (§4.1, Ap-
pendix F). A dense (semantic) realization of the
same policy, however, leaves the null unchanged
(Appendix Q), so the empty cell is not an arti-
fact of the deterministic matcher; the null holds
across random, structural, and dense retrieval. A
still-different retriever indexing a different corpus
could in principle relocate a model, and the re-
leased framework applies to such configurations
without modification, so we treat the result as a
controlled-policy data point rather than a closed
claim about retrieval-augmented drafting in gen-
eral.
Statistical adjudication.The 100-case cohort
trades raw nfor representativeness and probe
depth (§5.1); further batches can be drawn at zero
cost from the released sampling procedure (Ap-
pendix K). We report paired Wilcoxon signed-rank
tests with Holm–Bonferroni correction and boot-
strap 95% CIs on the headline ∆C5 and ∆C2 shifts,
together with TOST equivalence tests against the
pre-registered ±0.2 margin (Appendix O). No
∆C5 shift is significant after correction, and the
median ∆C5 is equivalent to zero within the pre-
registered margin for all four models; the GPT-4o-
mini boundary verdict is therefore supported by an
equivalence result rather than only by adjacency to
the threshold. Because per-case C5 varies widely
(±2to8; §6.4), the heavy-tailedmean ∆C5 is not
tightly bounded, so we frame the null as a state-
ment about the central tendency and rank distribu-
tion rather than the mean. The sample-size reversal
documented in §6.4 motivates the N=100 -with-
replicates sizing, which we flag so that downstream
users do not under-power their own runs.
Evaluation scope.A small fraction ( ∼10%) of
cases fail parsing due to OCR degradation or miss-
ing XML in older file wrappers; results are re-
ported stratified by amendment-size quartile (Ap-
pendix A.2) so that conclusions are conditional
on the parseable subset rather than confounded
by small-denominator cases. Prompts cap the ver-
batim examiner body at 8,000 characters, which
binds in 80% of cases (median body 13,448 char-acters); however, the structured rejection identifica-
tion (attacked claim numbers, statute section and
subsection, and cited references) precedes the body
and is never truncated, so the material the metric
scores against is retained in all 100 cases and a
statutory-rejection statement remains within the
retained window in every case. Per-model parse-
success is 97.4% overall (§6); recomputing the H1
verdicts on the common subset of 65 cases parsed
by all four models under every condition leaves
the prior-injection null unchanged (maximum ∆C5
= +0.10 ), and dropped cases do not differ sub-
stantially from the cohort in statute, amendment
pattern, or technology center (Appendix S). Finally,
our channels are designed to track attorney-practice
failure modes but are not a substitute for attorney
judgment; a sample-level attorney-validation study
(Appendix J) scores inter-rater agreement with the
automated channels.
Ethical and legal considerations
Patent claim amendment is a legally consequen-
tial act performed by registered agents. Our work
evaluates whether LLMs could in principle per-
form this task grounded-ly; we take no position
on whether they should be deployed for unassisted
drafting. We release the corpus and probes to en-
able scrutiny of commercial claims, not to rec-
ommend replacement of human counsel. Data
sources are public USPTO Open Data Portal ar-
tifacts with no copyright restriction. The corpus
does not contain personal information beyond the
inventor/attorney names that are part of the public
record. We do not re-identify or link across filings
beyond what ODP already exposes.
Use of AI assistants.We used commercial large
language models as the subjects of evaluation
(Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4,
GPT-4o-mini); their outputs constitute the exper-
imental data scored by our metric. AI assistants
were used during manuscript preparation for cod-
ing support, language and typo editing. It was not
used for research ideation, experimental design, or
the analysis itself.
Reproducibility Statement
All artifacts required to reproduce the headline
numbers are released under a permissive open
license at https://github.com/TeamLab/probi
ng-rag-patent-amendment:

•the parsed JSONL corpus of 7,385 four-tuples
and the parsing pipeline (§3, Appendix A);
•the deterministic cohort-selection procedure
with the batch 0 seed; additional replica-
tion batches are reproducible by re-running
select_cohort.py under any seed (Ap-
pendix N, Appendix K);
•the seven probe prompt templates and the five-
channel metric implementation (Appendix H,
Appendix I);
• the retrieval-pool index (Appendix N.7);
•the runnable model-call driver, the five-
channel scoring pipeline, and the analysis
scripts that regenerate every table from raw
model outputs, with pinned package versions
(requirements.txt).
Reproducing the model calls requires API access
to the four evaluated frontier LLMs; per-model
cost estimates and inference settings are reported
in Appendix B and §5.2.
Acknowledgments
This research was supported by the “Regional
Growth and Talent Development System (Anchor)”
Project, funded by the Ministry of Education and
Busan Metropolitan City (2026-Anchor-02-001-
004, 20%), and by the MSIT (Ministry of Science
and ICT), Korea, under grants through the National
Research Foundation of Korea (NRF) (No. RS-
2024-00354675, 40%) and the ICAN (ICT Chal-
lenge and Advanced Network of HRD) support
program supervised by the IITP (Institute for Infor-
mation & Communications Technology Planning &
Evaluation) (IITP-2023-RS-2023-00259806, 20%).
This work was also partly supported by the Tech-
nology Development Program (TIPS, RS-2024-
00554500, 10%) funded by the Ministry of SMEs
and Startups (MSS, Korea), and partly by the Korea
Institute of Marine Science & Technology Promo-
tion (KIMST) funded by the Ministry of Oceans
and Fisheries, Korea (RS-2026-25544055, 10%).
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.Donald T. Campbell and Donald W. Fiske. 1959.
Convergent and discriminant validation by the
multitrait-multimethod matrix.Psychological Bul-
letin, 56(2):81–105.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde de Oliveira Pinto, Jared Kaplan,
Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela
Mishkin, Brooke Chan, Scott Gray, and 39 others.
2021. Evaluating large language models trained on
code.Preprint, arXiv:2107.03374.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
RAG systems. InProceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval (SIGIR).
Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav
Goldberg. 2021. Amnesic probing: Behavioral expla-
nation with amnesic counterfactuals.Transactions of
the Association for Computational Linguistics, 9:160–
175.
Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. 2024. RAGAS: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of EACL (System Demonstrations).
Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiao-
jun Chen, and Ruifeng Xu. 2024. Enhancing noise
robustness of retrieval-augmented language models
with adaptive adversarial training. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (ACL).
Juraj Gottweis, Wei-Hung Weng, Alexander Daryin,
Tao Tu, Anil Palepu, Petar Sirkovic, Artiom
Myaskovsky, Felix Weissenberger, Keran Rong, Ryu-
taro Tanno, Khaled Saab, Dan Popovici, Jacob Blum,
Fan Zhang, Katherine Chou, Avinatan Hassidim, Bu-
rak Gokturk, Amin Vahdat, Pushmeet Kohli, and 15
others. 2026. Accelerating scientific discovery with
Co-Scientist.Nature.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat,
and Ming-Wei Chang. 2020. REALM: Retrieval-
augmented language model pre-training. InInterna-
tional Conference on Machine Learning.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research. ArXiv:2208.03299 (2022).
Yehoon Jang, Chaewon Lee, Hyun-seok Min, and
Sungchul Choi. 2025. PILOT-bench: A bench-
mark for legal reasoning in the patent domain with
IRAC-aligned classification tasks. InProceedings of
the Natural Legal Language Processing Workshop
(NLLP), at EMNLP.

Lekang Jiang, Pascal A. Scherz, and Stephan Goetz.
2025a. Patent-CR: A dataset for patent claim revi-
sion. InProceedings of NAACL (Long Papers).
Lekang Jiang, Pascal A. Scherz, and Stephan Goetz.
2025b. Towards better evaluation for generated
patent claims. InProceedings of ACL (Long Papers),
pages 3775–3788.
Carlos E. Jimenez, John Yang, Alexander Wettig,
Shunyu Yao, Kexin Pei, Ofir Press, and Karthik
Narasimhan. 2024. SWE-bench: Can language mod-
els resolve real-world GitHub issues? InInterna-
tional Conference on Learning Representations.
Seiya Kawano, Hirofumi Nonaka, and Koichiro
Yoshino. 2024. ClaimBrush: A novel framework
for automated patent claim refinement based on large
language models. InIEEE International Conference
on Big Data (BigData).
Valentin Knappich, Annemarie Friedrich, Anna Hätty,
and Simon Razniewski. 2025. PEDANTIC: A dataset
for the automatic examination of definiteness in
patent claims. InProceedings of PatentSemTech.
Jieh-Sheng Lee. 2023. Evaluating generative patent
language models.World Patent Information.
Jieh-Sheng Lee. 2024. InstructPatentGPT: Training
patent language models to follow instructions with
human feedback.Preprint, arXiv:2406.16897.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Informa-
tion Processing Systems.
Hyunseung Lim, Sooyohn Nam, Sungmin Na, Ji Yong
Cho, June Yong Yang, Hyungyu Shin, Yoonjoo Lee,
Juho Kim, Moontae Lee, and Hwajung Hong. 2025.
PANORAMA: A dataset and benchmarks capturing
decision trails and rationales in patent examination.
InAdvances in Neural Information Processing Sys-
tems (NeurIPS).
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12.
Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Ya-
mada, Shengran Hu, Jakob Foerster, David Ha, and
Jeff Clune. 2026. Towards end-to-end automation of
AI research.Nature.
Marco Túlio Ribeiro, Tongshuang Wu, Carlos Guestrin,
and Sameer Singh. 2020. Beyond accuracy: Be-
havioral testing of NLP models with CheckList. In
Proceedings of ACL. ACL 2020 Best Paper.Eva Sharma, Chen Li, and Lu Wang. 2019. BIG-
PATENT: A large-scale dataset for abstractive and
coherent summarization. InProceedings of ACL.
Yaorui Shi, Sihang Li, Taiyan Zhang, Xi Fang, Jiankun
Wang, Zhiyuan Liu, Guojiang Zhao, Zhengdan Zhu,
Zhifeng Gao, Renxin Zhong, Linfeng Zhang, Guolin
Ke, Weinan E, Hengxing Cai, and Xiang Wang. 2025.
Intelligent system for automated molecular patent in-
fringement assessment.Preprint, arXiv:2412.07819.
Homaira Huda Shomee, Suman Kalyan Maity, and
Sourav Medya. 2025. PATENTWRITER: A bench-
marking study for patent drafting with LLMs.
Preprint, arXiv:2507.22387.
Yiwen Song, Yale Song, Tomas Pfister, and Jin-
sung Yoon. 2026. PaperOrchestra: A multi-agent
framework for automated AI research paper writing.
Preprint, arXiv:2604.05018.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InNeurIPS Datasets
and Benchmarks Track.
U.S. Patent and Trademark Office. 2025. Patents
dashboard. https://www.uspto.gov/dashboard/
patents/.
U.S. Patent and Trademark Office, Office of the
Chief Economist. 2017. Office action re-
search dataset for patents. USPTO Open
Data. https://www.uspto.gov/ip-policy/econ
omic-research/research-datasets.
Emiel van Miltenburg, Chris van der Lee, and Emiel
Krahmer. 2021. Preregistering NLP research. InPro-
ceedings of NAACL. NAACL 2021 Best Thematic
Paper.
Richard Van Noorden. 2014. Publishers withdraw more
than 120 gibberish papers.Nature News.
Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov,
Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart
Shieber. 2020. Investigating gender bias in language
models using causal mediation analysis. InAdvances
in Neural Information Processing Systems.
Qiyao Wang, Shiwen Ni, Huaren Liu, Guhong Chen,
Xi Feng, Chi Wei, Qiang Qu, Hamid Alinejad-Rokny,
Yuan Lin, and Min Yang. 2024. AutoPatent: A multi-
agent framework for automatic patent generation.
Preprint, arXiv:2412.09796.
Yongmin Yoo, Qiongkai Xu, and Longbing Cao. 2025.
PatentScore: Multi-dimensional evaluation of LLM-
generated patent claims. InProceedings of EMNLP.
Shuyang Yu, Jianan Liang, and Hui Hu. 2025. ToC:
Tree-of-claims search with multi-agent language
models.Preprint, arXiv:2511.16972.

A Corpus construction pipeline
A.1 Source pool and enrichment chain
We start from the PILOT-Bench (Jang et al., 2025)
PTAB-appeal subset of 13,749 proceedings, each
carrying a patent application number, decision out-
come, statute citations, and technology-center as-
signment. For each proceeding we run a five-
stage enrichment against the USPTO Open Data
Portal (ODP): (i) resolve the application number
via ODP query and obtain the file-wrapper docu-
ment list; (ii) identify the first non-final rejection
(USPTO code CTNF, “correspondence: notice of
non-final rejection”), the immediately preceding
incoming Claims filing (pre-amendment), and the
first incoming Claims filing after the CTNF (post-
amendment); (iii) download each as XML, exclud-
ing filings available only as PDF/image scans; (iv)
parse XML under three schema families (§A.2); (v)
compute per-case diff with status ∈{kept, modified,
new, cancelled} via claim-number alignment, plus
SequenceMatcher similarity for modified claims.
A.2 XML schema handling
Three schema families coexist in ODP’s claim doc-
uments:
•Legacy DTD <us-patent-application>
for pre-2014 filings. Claim text is in-
line; amendments use [[deleted]] bracket
markup. Our parser splits claim bodies on
numeric headers and extracts bracket spans.
•USPTO ClaimsDocument v1.3
<pat:ClaimsDocument> for mid-generation
filings. Uses <pat:Claim><pat:ClaimText>
wrappers with underline and
<pat:DeletedText>markup.
•ST96 v2 <uspat:ClaimsDocument> for
recent filings. Dual URI namespace;
<pat:Ins> /<pat:Del> insertion and deletion
tags; <ImplicitClaim> elements encode
claim-number ranges that expand into sepa-
rate records.
Rejection documents use
<uspat:OutgoingDocument> with struc-
tured <uscom:FormParagraph> and
<uscom:DataField> slots encoding claim
numbers, statute subsection, rejection type, and
cited references. V7.1 documents replace semantic
FormParagraphNumber identifiers with generic
IDs; for those we fall back to regular-expressionparsing of the prose body. OCR-confidence
wrappers ( <pat:OCRConfidenceData> and the
legacy <confidence> ) are stripped before parsing.
A.3 Parser hardening
A targeted hardening pass on 2026-04-18 raised
cohort coverage from 76/100 to 100/100 via four
sequential fixes: dual-URI namespace fallback
plus <ImplicitClaim> range expansion (93/100);
inline-header fallback with paren-form and multi-
inline-text wrapper splits (97/100); bracket [Claim
N]and word-prefix Claim N (Status): matching
with DTD legacy inline fallback (99/100); and a
deterministic case replacement (Appendix N). The
same hardening applied to the full βcorpus re-
covered several thousand additional cases that the
baseline parser had silently dropped.
A.4 Final corpus statistics
Funnel from PILOT-Bench source pool to release
corpus and cohort eligibility:
•PILOT-Bench PTAB-appeal subset: 13,749
proceedings.
•ODP retrieval targets (resolvable application
numbers): 9,956.
•Parseable (pre, post, CTNF) triples: 7,385
(74.2% parse yield).
• Case-level C2 computable: 5,755.
•Cohort eligibility pool (after parse-health, axis
coverage, and data-defect filters): 4,270.
The remaining 2,571 parse failures concentrate
in documents without XML download options
(older filings), publications not found in ODP, and
a small set of malformed wrappers that survive
parser hardening. Across the full 7,385-case cor-
pus: per-claim status modified 99,335 (64.5%),
cancelled 33,118 (21.5%), new 17,008 (11.0%),
kept 4,459 (2.9%). Modified-claim similarity ra-
tio median 0.933, mean 0.841, Q1 0.827, Q3
0.966 ( n= 99,335 ). Case-level pre-to-post Lev-
enshtein similarity median 0.72, Q1 0.55, Q3 0.84
(n= 5,755 ). First-rejection statute distribution:
§102 2,337 (31.6%), §112 1,245 (16.9%), §101
950 (12.9%), §103 738 (10.0%). XML format mix:
DTD legacy 3,576 (48.4%), namespaced (v1.3 plus
ST96) 3,808 (51.6%). Coarse amendment pat-
tern: modify_only 5,043 (68.3%), cancel_heavy
1,410 (19.1%), other 757 (10.3%), new_or_add
175 (2.4%).

A.5 Relation to other corpora
The corpus is constructed to enable grounded-
revision evaluation (pre ↔rejection ↔prior art
↔post) rather than to maximize raw scale. As
noted in §3.3, ours is the only publicly described
corpus that aligns rejection context, cited prior
art, pre/post claim pair, and amendment diff at
the XML level. Patent-CR provides the pre/post
pair only; PANORAMA, PEDANTIC, and PILOT-
Bench target judgment, classification, or upstream
retrieval rather than applicant-side amendment.
B Per-model cost estimates
Per-call token counts and vendor pricing yield ap-
proximately $110 for Stage 1 (Claude Sonnet 4 and
GPT-5.4 across the full probe battery on 100 cases
×8 conditions ×3 replicates) and approximately
$110 for Stage 2 (Claude Haiku 4.5 and GPT-4o-
mini under the same protocol), for a combined ex-
perimental cost of approximately $220 across 9,600
model calls. Infrastructure cost is zero: USPTO
ODP is a public API and all compute is local.
C Worked examples per channel
This appendix maps each channel’s numeric out-
put to a concrete within-case contrast drawn from
the main experiment. Scores are illustrative; they
show what each channel responds to, not where
population medians sit.
C.1 C1 Grounding alignment
Case.PILOT-2019005776 (Reversed, §101-
related, TC 2100, 2020–25, cancel_heavy).Model.
GPT-4o-mini. Under baseline the model’s modified
limitations overlapped the rejected limitations at
rate0.014 (edit off-target). Under Probe A (Claim
Truncation) the overlap rose to 0.402 within-case,
an order-of-magnitude shift on the same case and
model. C1 catches whether an amendment lands on
the examiner-attacked limitation rather than else-
where in the claim.
C.2 C2 Revision Locality
Case.PILOT-2020002555 (Reversed, §103+112,
TC other, 2020–25, cancel_heavy).Model.GPT-
4o-mini. The gold amendment has edit distance
591 characters. Under baseline C2 = 0.26 (26% of
gold scale, an under-edit). Under Probe F (Random
Retrieval) C2 = 7.40 (over 7×gold scale, a near-
complete rewrite). The channel registers the twomisses as categorically different rather than as a
single failure mode.
C.3 C3 Scope preservation
Case.PILOT-2020003998 (AIP, §103+112, TC
3700, 2020–25, cancel_heavy).Model.Claude
Haiku 4.5. Under baseline noun-phrase Jaccard
with the invention core is 0.983 . Under Probe D
(Boilerplate Injection) it collapses to 0.073 as the
injected canonical phrases overwrite the invention-
core noun phrases. This is the over-narrowing /
redirection failure mode C3 targets.
C.4 C5 Template dependence
Case.PILOT-2013007708 (Affirmed, §103, TC
3700, 2010–14, modify_only).Model.GPT-4o-
mini. Under baseline C5 = 3.34 canonical-phrase
hits per 1,000 characters (7 hits in 2,081 characters).
Under Probe G (Structural Retrieval) C5 = 11.66
(49 hits in 4,181 characters), a 3.5× increase. This
is the caseγreferenced in Appendix D.
C.5 Simpler-baseline comparison
On each worked example,
BLEU/ROUGE/BERTScore baselines fail to
register the within-case direction our channels
detect. On PILOT-2020002555, BLEU-4 scores
baseline and F as equally distant from gold ( ∼0.12
vs∼0.10), while C2 reads them as 0.26 vs7.40.
On PILOT-2020003998, BERTScore-F1 moves
from 0.91 to 0.73 under Probe D, but the magnitude
does not flag the invention-core-overwrite failure
mode; C3 moves 0.98 to0.07 and registers the
scope collapse directly. On PILOT-2013007708,
raw n-gram recurrence rises 1.2× baseline
(domain-natural repetition) while C5 rises 3.5× ,
isolating retrieval-induced template reuse from
intrinsic patent prose repetition.
C.6 Caseβ: GPT-5.4 F-vs-G sign flip
Case.PILOT-2020006264 (Affirmed, §101-
related, TC 3600, modify_only).Model.GPT-5.4.
C5 baseline 11.3 moves to 13.5 under F ( ∆F=
+2.25 , inflation) and 9.8 under G ( ∆G=−1.50 ,
suppression). Same input, same retrieval pool,
same k; only the selection rule differs, yet the effect
inverts in sign. The other three models on this case:
Sonnet 4 near-flat ( ∆F= +0.01 ,∆G=−0.14 );
Haiku 4.5 mildly inflates on F ( +1.96 ) but is flat
on G (−0.29 ); GPT-4o-mini consistently reduces
(∆F=−3.20 ,∆G=−3.32 ). This case con-
cretely realizes the H2-rejection verdict in §6.1:

for GPT-5.4, random versus structural retrieval
changes not just the magnitude but the sign of the
template-dependence effect.
D Sample-size reversal
The Table 1 verdicts are computed at N=300 per
cell (100 cases ×3 replicates). A preliminary
smoke run at N=10 per cell, on identical inputs
and the same scoring code, gave ∆C5= + 0.84
for GPT-5.4 under F, a magnitude that would have
placed GPT-5.4 firmly in the H1-supported cell.
AtN=300 the same effect collapses to −0.12 , a
complete sign reversal. The reversal is sampling
variability on a 100-case probe-based cohort used
atN=10 , not a calibration issue, and is consistent
with the per-case dispersion of ±2to8that §6.1
reports on this channel.
We document the GPT-5.4/F cell explicitly be-
cause it was the cell that triggered the N=10→
N=300 escalation decision and is the largest docu-
mented reversal in our pipeline; the smoke run did
not include a systematic N=10 replication on every
(model,condition) cell in Table 1, so we report this
one reversal as a benchmark-methodology finding
rather than as a cell-by-cell stability sweep. The
implication for downstream users is the same in
either case: replications should be sized at or above
the pre-registered N=100 per condition with repli-
cates rather than relying on smoke-run effect sizes,
because at N=10 a sign-reversing reversal is plau-
sible on any single cell of a 100-case probe-based
cohort.
EProbe-D anchor: per-model breakdown
Extending the cross-model comparison in §6.3,
the per-model |∆F|-vs-|∆D|gaps on three of four
models are within 0.13 (the maximum observed
gap on those three models, reported descriptively as
a post-hoc magnitude rather than as a pre-registered
tolerance), indicating that random retrieval does
not behave qualitatively differently from canoni-
cal boilerplate on Sonnet 4, GPT-5.4, and GPT-
4o-mini. Haiku 4.5 is the per-model exception:
its∆F= +0.19 exceeds |∆D|= 0.09 in abso-
lute magnitude, consistent with its cell-(d) “inert
/ mild prior-injector” assignment, where retrieval
produces the largest random-F template inflation in
the panel (though still below the +0.2 H1 thresh-
old). Even on Haiku 4.5, the retrieval shift is
the same order as the boilerplate-prepend dosage.
The cross-model qualitative reading therefore stillTable 4: Per-model probe ×channel deltas. Conditions
A–E are observational; F and G are retrieval interven-
tions. Leading zeros omitted in baselines for compact-
ness.
cond∆C1∆C2∆C3∆C5
Sonnet 4(base: C1 .009, C2 .617, C3 .887, C5 3.99)
A+0.004−0.083−0.005 +0.01
B+0.001−0.038−0.018 +0.08
C−0.001 +0.001−0.006−0.09
D+0.001 +0.047−0.026 +0.03
E+0.001−0.088−0.002−0.30
F−0.001−0.074−0.004−0.16
G−0.001−0.013−0.007−0.13
Haiku 4.5(base: C1 .016, C2 1.147, C3 .738, C5 3.40)
A+0.013−0.090 +0.001 +0.16
B−0.001−0.035−0.004 +0.16
C+0.003−0.103−0.007 +0.03
D−0.000 +0.032−0.017−0.09
E−0.003−0.060 +0.002 +0.16
F−0.001−0.174 +0.027 +0.19
G−0.002−0.189 +0.038 +0.03
GPT-5.4(base: C1 .008, C2 .789, C3 .892, C5 3.73)
A+0.005−0.035 +0.003 +0.19
B+0.001−0.087 +0.003 +0.06
C−0.000−0.038 +0.009 +0.10
D−0.000 +0.051−0.003 +0.07
E−0.002−0.099 +0.020 +0.06
F+0.000 +0.016−0.001−0.12
G+0.002 +0.058 +0.002 +0.06
GPT-4o-mini(base: C1 .017, C2 1.459, C3 .798, C5 3.14)
A+0.018−0.035−0.007 +0.02
B−0.009−0.050−0.030 +0.08
C−0.001−0.047 +0.023−0.29
D−0.009−0.044 +0.034−0.07
E−0.007−0.008 +0.028−0.07
F−0.006−0.044 +0.021 +0.20
G−0.006−0.076 +0.028−0.27
holds: random retrieval is bounded by a generic
prepended-context effect, with per-model magni-
tudes reported individually rather than averaged.
F Probe x channel full table
Medians across 100 cohort cases ×3 replicates
per (model, condition). Channels C1, C2, C3, and
C5 are shown; C4 is a directional-hit-rate mea-
sure with paired-probe structure, reported sepa-
rately in Appendix G. Probe G uses a determin-
istic feature-based matcher rather than a dense re-
triever so that the F vs G contrast is interpretable
as a structural-similarity test rather than a retriever-
quality test. An embedding retriever would con-
flate the structural-similarity signal we want to
isolate with the representation-learning quirks of
whichever embedder is chosen.

Table 5: C4 directional hit-rate per (probe, channel) pair,
pooled across four models and three replicates on the
100-case cohort. FLAT predictions use a |∆|<0.05
tolerance.
(probe, channel) predicted hit-raten
B, C1 FLAT0.95 382
B, C3 FLAT0.64 387
F, C5 UP0.50 383
G, C5 UP0.47 384
D, C5 UP0.42 392
Cmechtrue , C1 UP0.40 178
Cdecoy, C1 DOWN0.32 170
grand C40.56 2,276
G C4 robustness per probe-pair
Directional hit-rate per (probe, channel) pair,
pooled across the four models (Claude Sonnet 4,
Claude Haiku 4.5, GPT-5.4, GPT-4o-mini) and
three replicates on the 100-case cohort. Each case-
model contrast scores 1 if the per-case ∆from
baseline matches the pre-registered direction (UP /
DOWN / FLAT with |∆|<0.05 threshold), 0 oth-
erwise. The grand C4 in Table 5 is the unweighted
mean over all hits, 0.56 onn=2,276 case-model
contrasts. C subconditions ( Cdecoy,Cmechtrue ) are
split per the probe-C index (Appendix N.7); 10 of
100 cohort cases are skipped at probe-build time,
so the C-rownis smaller than the others.
H Probe prompt templates
The shared system prompt and base user-prompt
template are released in contexts/system.txt
and contexts/user_base.txt . Probe-
specific perturbations (A–G) are applied by
scripts/build_probe_prompts.py : the Probe
D boilerplate block, the Probe E drafting hints,
and the Probe F/G retrieval-insertion templates are
defined inline in that script.
I Five-channel metric computation
Reference implementations for channels C1
through C5 are provided in the supplementary
material ( scripts/compute_c{1,2,3,4,5}.py ).
Each channel is a deterministic function of gen-
erated text, parser output, and the XML gold post-
amendment claim, as defined in §4.2.
J Attorney validation study
A 30–50-case double-rated sample assessing inter-
rater agreement between our automated channels
and licensed patent attorneys is committed post-
review. Reported metrics will include Cohen’s κfor categorical channels and Spearman ρfor con-
tinuous channels, against the automated scoring
on the same cases. A 10-case pre-release sanity
sample showed all four channels moving in the di-
rection the reading attorney predicted; the full κ
quantification is reserved for the published study.
The 10 cases used in the sanity sample are
the same 10 cases across all four channels (a
single shared subset, not a per-channel subset),
drawn deterministically from the 100-case cohort
by application-number ordering, so that each case
contributes one C1, C2, C3, and C5 reading to the
attorney’s directional check. Application-number
ordering correlates loosely with filing date and
therefore with XML-format era, so the 10-case
sanity sample is not stratum-balanced; the full 30–
50-case study will instead draw a stratum-balanced
sample across the six cohort axes so that the direc-
tional check is not era-biased. The reading attor-
ney was blind to the automated scores at the time
of directional prediction: per-case attorney read-
ings were elicited from the input materials (pre-
amendment claim, rejection rationale, cited prior
art, and the model-generated amendment) before
the automated scores were revealed; the compari-
son was then made between the attorney’s predicted
direction and the automated direction. The full κ
study will pre-register the same blinding protocol
on the larger 30–50-case sample, with double-rated
readings to estimate inter-rater reliability alongside
agreement with the automated channels.
K Batch replication protocol
Sampling procedure.Deterministic cohort selec-
tion ( scripts/select_cohort.py ) under the six-
axis marginal-match procedure described in §5.1
and Appendix N. The procedure is reproducible
from seed.
Batch 0.The seed used for primary analy-
sis is 42; cohort case identifiers are released as
cohort_batch0.json.
Additional batches.All results in this paper
are reported on batch 0. Because the selection
procedure is deterministic given a seed, any ad-
ditional 100-case cohort can be regenerated with
select_cohort.py under a new seed. We run
a single well-constructed cohort by design: pre-
emptive multi-batch analysis would either double
the budget without informing the primary hypothe-
ses or, if abbreviated, weaken probe depth.

K.1 Stage 2 escalation criteria
Stage 2 (Claude Haiku 4.5 and GPT-4o-mini on
the full probe battery) is performed when at least
two of the following hold consistently across both
Stage-1 flagships: (i) Probe F yields ∆C5>0.2
relative to no-retrieval baseline; (ii) Probes F and
G yield comparable C5 shifts ( |∆F−∆G|<0.1 ,
same sign); (iii) retrieval does not reduce C5 below
baseline under at least one of F or G. When fewer
than two conditions hold, Stage 1 already answers
the question asymmetrically across vendor families,
and Stage 2 primarily clarifies scope rather than
shifting conclusions.
L Stratified results
Per-stratum result tables (by statute section, tech-
nology center, XML format, and amendment pat-
tern class) are released as supplementary TSVs
alongside the corpus and parsing code. The co-
hort’s six-axis marginal match (Appendix N) en-
sures each stratum has at least five cases, support-
ing within-stratum effect-size estimation.
M Corpus release index
The released corpus consists of: the parsed
JSONL of 7,385 four-tuples; the parsing code;
the application-number index permitting zero-
cost reconstruction from USPTO ODP; and
cohort_batch0.json (the batch 0 case identi-
fiers); additional batches are regenerable from
select_cohort.py under any seed. All released
under a permissive open license; USPTO ODP data
has no copyright restriction.
N Cohort construction
N.1 Why 100 cases
The cohort size is the smallest nthat simultane-
ously satisfies four constraints: (i) within-model
paired comparisons with >80% power at Cohen’s
d= 0.3 ,α= 0.05 ; (ii) six-axis marginal-match
feasibility with minimum cell size 5 (requires
n≥∼ 90); (iii) probe-depth budget at 7 probes ×
3 reps×4 models = 84 calls per case (8,400 calls
before ablations) which is feasible at n= 100 but
triples at n= 300 without adding decisive within-
model power; (iv) attorney-validation headroom,
since a 30–50-case audit is a meaningful fraction
ofn= 100but only 3% ofn= 1,000.N.2 Why stratified, not random
A random 100-case draw from the 4,270-case eligi-
bility pool would under-represent the rejection out-
comes most informative for H2 contrasts (Reversed
cells at ∼30% raw frequency), rare statutes (§101-
related and §112-alone at ∼6–10% each, falling be-
low the cell-minimum 5 threshold with non-trivial
probability), and minority technology centers (long
tail of 1–3-case bins that cannot support per-TC
stratified robustness). Stratified sampling with tar-
get floors guarantees presence on all six axes.
N.3 Sampling procedure
Iterative marginal matching, not cross-product enu-
meration. The six-axis cross-product has too many
empty cells at n= 100 ; iterative matching instead
scores each case in the pool by the rarity of its
six-axis vector relative to the targets, greedily ac-
cepts cases that move the cohort toward all six tar-
get marginals simultaneously, and uses the seed to
break ties. At each step the procedure checks cell-
minimum m= 5 for every bin of every axis. Code:
scripts/select_cohort.py . Seed for batch 0:
42.
N.4 Eligibility filter
The 4,270-case pool is the subset of the 7,385 β-
parsed corpus satisfying: (i) parse health (pre- and
post-claims parseable, at least one CTNF rejection
instance, statute section identifiable on the first re-
jection); (ii) axis coverage (subdecision outcome
∈{Affirmed, Reversed, Affirmed-in-Part}, exclud-
ing dismissed and remanded outcomes that appear
in∼8% of the source pool); (iii) no obvious data
defect (pre-claim 1 ≥50 characters, at least one
modified or cancelled per-claim action so that there
is something to score).
N.5 Batch 0: achieved vs. target
Marginal match per axis: outcome 40/40 Af-
firmed, 40/40 Reversed, 20/20 AIP. Statute
35/20/15/15/10/5 target vs. 36/20/15/15/9/5
achieved across 103-alone, 102+103, 103+112,
101-related, 112-alone, and other. Technology
center 20/17/14/12/12 across the top six TCs (exact
match). Pattern modify_only 45 / cancel_heavy
20 / mixed 15 / new_or_add 10 / other 10 (exact
match). Format dtd_legacy 25 / ns_claims 75
(exact match). Year 25/45/30 target vs. 24/46/30
achieved across 2010–14, 2015–19, 2020–25.
All bins pass the m= 5 minimum (smallest bin:
statute "other" at 5).

Table 6: C5 baseline-versus-retrieval tests. No com-
parison is significant uncorrected (0 of 8), and none
survives Holm correction. Median-based TOST estab-
lishes equivalence to zero within the ±0.2 margin for
every model under both probes.
Model med∆Fp F med∆Gp G
Claude Sonnet 4+0.000.52+0.000.78
Claude Haiku 4.5+0.070.11+0.010.33
GPT-5.4+0.000.14+0.010.30
GPT-4o-mini+0.000.87+0.000.78
N.6 Post-registration amendments
Cohort.PILOT-2019006614 was dropped because
itsβpre_clm XML contains two <Claim> wrap-
pers with eleven unnumbered <ClaimText> ele-
ments and no inline-header pattern, making pre-
amendment text not deterministically recoverable.
A pre-declared replacement rule selected the low-
est app_num case in the eligibility pool match-
ing the dropped case’s six-axis vector (Affirmed /
§101-related / TC 3600 / 2020–25 / new_or_add /
ns_claims) with parse-success under the patched
parser. Selected: PILOT-2022000316. Marginal
impact: zero.
Model matrix.GPT-4o-mini replaces GPT-5-
mini in the smaller-OpenAI cell. Trigger: in the
N=10 multi-model smoke run, GPT-5-mini pro-
duced 25 of 30 zero-parseable-claim responses, 13
of 30 max_tokens truncations, and did not accept
the pre-registered temperature 0.3. A pre-declared
fallback rule substitutes to the next stable smaller
OpenAI model supporting temperature control and
canonical claim format. Verification on GPT-4o-
mini: 29 of 30 parseable responses, 0 of 30 trunca-
tions, temperature 0.3 accepted.
N.7 Retrieval-pool disjointness
The invariant cohort ∩pool=∅holds by construc-
tion, since the pool is the βcorpus with the 100
cohort cases removed.
O Significance tests and equivalence
For each model we aggregate the three replicates to
a per-case mean, pair the retrieval condition against
the no-retrieval baseline case-by-case, and apply
a paired Wilcoxon signed-rank test. p-values are
Holm–Bonferroni corrected within the C5 test fam-
ily; bootstrap 95% CIs on the median paired differ-
ence use 10,000 resamples (seed 42). Equivalence
against the pre-registered±0.2margin is assessed
by two one-sided tests (TOST) on the median ∆C5.
On C2 the same battery finds 3 of 8 comparisonsTable 7: Retrieval quality, F vs G vs dense-G, over each
condition’s actual picks (300 per condition).
Condition mean cosine struct. (0–3) statute match
random-F 0.587 0.746 31.7%
structural-G 0.607 2.342 90.0%
dense-G 0.871 1.173 55.3%
nominally significant (Haiku 4.5 F p=0.008 and
Gp=0.036 ; Sonnet 4 F p=0.049 ), all decreases,
none surviving Holm correction. The effect is car-
ried by Haiku 4.5 rather than a Claude-versus-GPT
boundary (§6.2).
P Paraphrase-sensitive (semantic) C1
To check that the grounding null is not an arti-
fact of lexical (trigram) overlap, we recompute
C1 with a paraphrase-sensitive embedding scorer
(all-MiniLM-L6-v2 ), scoring the same generated
and rejected limitations by embedding cosine rather
than token overlap. Lexical and semantic C1 are
only loosely concordant (Spearman ρ= 0.37 )
and flag different cases, so the semantic variant
is a genuine second measurement rather than a
re-derivation. The retrieval null is unchanged: 0
of 8 baseline-versus-retrieval comparisons survive
Holm correction under semantic C1, and the two
uncorrected near-misses (Haiku 4.5 and GPT-4o-
mini under F) aredecreases, not the grounding
gains the optimistic account predicts.
Q Dense-retriever realization of Probe G
Dense-G is a controlled swap of the retrieval mech-
anism only: an embedding-cosine top-3 retriever
(all-MiniLM-L6-v2 over rejection-plus-attacked-
claim text) replaces Probe G’s deterministic feature
matcher, holding the pool, k=3, exemplar fields,
injection format, and base prompt byte-identical to
F and G. A shared-space retrieval-quality log (Ta-
ble 7) confirms the three conditions are genuinely
different retrievers.
Dense-G retrieves far more semantically similar
exemplars than F or G (cosine 0.87 vs ∼0.60) while
its structural-tag overlap sits between them, so it is
a real third mechanism rather than a re-derivation of
G. Generation ran on all four models (1,200 calls,
0 failures, $18.32). Scored with the same deter-
ministic pipeline against a Holm-corrected 16-test
family (four models ×C5, C2, lexical C1, semantic
C1), 0 of 16 comparisons survive correction and no
model shows a significant C5 increase; the closest
is Haiku 4.5 on C5 ( p=0.053 , uncorrected). The

Table 8: C5 median by retrieval depth, random-F |
structural-G.
Modelk=1k=3k=5k=10
Sonnet 4 4.90/4.94 4.94/4.94 4.94/4.92 5.00/4.94
Haiku 4.5 4.15/4.29 4.29/4.27 4.11/4.37 4.32/4.41
GPT-5.4 4.93/4.85 4.94/4.83 4.93/4.88 4.85/4.92
GPT-4o-mini 4.12/3.95 4.27/4.06 4.05/4.10 4.04/3.94
null is thus robust to how the retriever is built, not
only to which fixed matcher is chosen.
R Retrieval-depth sensitivity
k=3 was a fixed, untuned default balancing exem-
plar diversity against prompt length and inference
cost. Varying only k∈ {1,3,5,10} (cohort, re-
triever, scaffold, exemplar ordering, model, and
decoding held fixed), C5 medians (random-F |
structural-G) are given in Table 8.
C5 shows no monotonic increase with k. Quan-
tifying depth variation as the maximum deviation
from the k=3 value, this is ≤0.18 for seven of the
eight model ×condition series; the one larger case
(GPT-4o-mini, random-F, 0.23) is non-monotonic,
peaking at k=3 and lower at k=5 andk=10 . C1 is
practically stable across depths (only Sonnet 4 dips
slightly at k=10 ); C3 rises mildly with kfor the
Claude models and is flat for GPT; C2 is noisy with
no monotonic trend. Applying the pre-registered
criteria at every depth yields the same qualitative
H1–H3 verdicts, so the conclusions are not artifacts
ofk=3. Prompt length grows with kas expected
(approximate median 4.5k tokens at k=1 to 6.2k at
k=5,cl100k tokenizer), and the office-action trun-
cation rate (8,000-character cap on the rejection
text) is 80% and constant across depths.
S Truncation and parse robustness
The examiner rejection is presented to the model
as a set of structured rejection instances (attacked
claim numbers, statute section and subsection, re-
jection type, and cited references; §4.1) followed
by the verbatim examiner body, and only the body
is capped at 8,000 characters (Appendix R). Across
the 100-case cohort the footer-trimmed body ex-
ceeds the cap in 80% of cases (median body length
13,448 characters, range 2,333–41,801), so most
prompts truncate the tail of the verbatim reason-
ing. The truncation does not, however, remove
the material the metric scores against: the struc-
tured rejection instances precede the body and are
never clipped, so the attacked-claim identification,
statutory basis, and cited references are present forall 100 cases, and a statutory-rejection statement
(“. . . rejected under 35 U.S.C. . . . ”) remains within
the retained window for all 100 cases. C1 aligns
the generated amendment against the claims named
as rejected, which are carried by the structured in-
stances; truncation therefore clips only the tail of
the examiner’s verbatim prose, not the identifica-
tion of the attacked limitation.
Differing per-model subsets.A generated
amendment occasionally fails to parse and is
dropped, so each model is scored on a slightly dif-
ferent subset. Per-model parse-success is 97.4%
overall (§6). Because failures fall on different
cases per model, we recompute the H1 criterion on
the common subset of 65 cases parsed by all four
models under every condition: no model reaches
the∆C5≥0.2 threshold (maximum +0.10 for
GPT-4o-mini), so the prior-injection verdict is un-
changed when every model is scored on identical
cases. Dropped cases track the cohort’s marginals
rather than concentrating in a stratum: by first-
rejection statute, §103 is 40% of failures vs. 36%
of the cohort; by amendment pattern, modify-only
is 40% vs. 45%; and the technology-center distri-
bution is comparable (largest shift: art unit 1700 at
25% of failures vs. 14% of the cohort).
Complete-rejection subset.On the 20 cases
whose rejection body is not truncated, per-case C5
estimates are dominated by small-sample variance,
consistent with the N-sensitivity documented in
§6.4 (where a N=10 estimate of +0.84 reversed
to−0.12 atN=300 ); we therefore do not adju-
dicate H1 on this 20-case subset. Because trun-
cation leaves the scored attacked-limitation iden-
tification intact, there is no mechanism by which
it would bias the verdict, and the common-subset
check above provides the properly-powered robust-
ness result.