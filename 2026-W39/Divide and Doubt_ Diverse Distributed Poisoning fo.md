# Divide and Doubt: Diverse Distributed Poisoning for Retrieval-Augmented Generation

**Authors**: Tianhao Chen, Yuhan Wei, Weifei Jin, Zhengyuan Jiang, Yuepeng Hu, Neil Zhenqiang Gong

**Published**: 2026-09-22 21:42:19

**PDF URL**: [https://arxiv.org/pdf/2609.27090v1](https://arxiv.org/pdf/2609.27090v1)

## Abstract
Multi-passage corpus poisoning often repeats one target claim across similar documents, creating correlated lexical and semantic patterns that similarity- and conflict-aware defenses can suppress jointly. We introduce DnD (Divide and Doubt), a targeted attack based on two principles: distributing support for the target answer across stylistically diverse passages, and including a passage that casts doubt on evidence for the reference answer. The first disperses poison-passage representations in embedding space, while the second strengthens target adoption when multiple poisoned passages are retrieved. We evaluate DnD on two open-domain QA datasets across three LLMs and nine RAG configurations, under both black-box and white-box access to the retriever. Across these settings, DnD matches or outperforms prior attacks in most configurations, with its largest gains against clustering- and conflict-aware defenses.

## Full Text


<!-- PDF content starts -->

Divide and Doubt: Diverse Distributed Poisoning for
Retrieval-Augmented Generation
Tianhao Chen∗Yuhan Wei∗Weifei Jin Zhengyuan Jiang
Yuepeng Hu Neil Zhenqiang Gong
Duke University
Abstract
Multi-passage corpus poisoning often repeats one target claim across similar docu-
ments, creatingcorrelatedlexicalandsemanticpatternsthatsimilarity-andconflict-aware
defenses can suppress jointly. We introduceDnD(Divide and Doubt), a targeted attack
based on two principles: distributing support for the target answer across stylistically
diverse passages, and including a passage that casts doubt on evidence for the reference
answer. The first disperses poison-passage representations in embedding space, while
the second strengthens target adoption when multiple poisoned passages are retrieved.
We evaluateDnDon two open-domain QA datasets across three LLMs and nine RAG
configurations, under both black-box and white-box access to the retriever. Across these
settings,DnDmatches or outperforms prior attacks in most configurations, with its
largest gains against clustering- and conflict-aware defenses.
1 Introduction
Retrieval-augmented generation (RAG) grounds model outputs in passages retrieved from an external
corpus (Lewis et al., 2020). Under a corpus-write threat model, however, this external memory also
becomes an attack surface. Retriever-focused poisoning can optimize injected passages to enter the
retrieved context (Zhong et al., 2023), while PoisonedRAG showed that a small number of malicious
passages can induce an attacker-selected answer by satisfying separate retrieval and generation
conditions (Zou et al., 2025). When the attacker can inject multiple passages, the problem is no
longer only whether each passage is individually effective, but also how the passages interact as a set.
A straightforward multi-passage attack repeatedly supports the same malicious answer. Because
all passages are generated for the same question–target pair, they can converge on similar document
styles, contextual framings, and wording even when their sentences are paraphrased. These correlated
stylistic, semantic, and lexical patterns are particularly problematic against defenses that analyze
retrieved evidence jointly. TrustRAG and SeCon-RAG, for example, use similarity-, clustering-, and
conflict-based mechanisms to identify or discount suspicious evidence (Zhou et al., 2025; Si et al.,
2025). Surface paraphrasing alone cannot address repeated support rationales, while unconstrained
∗These authors contributed equally.
1
arXiv:2609.27090v1  [cs.CR]  22 Sep 2026

variation may weaken support for the target answer or introduce contradictions. The central challenge
is therefore to diversify the evidence presented across passages while preserving agreement on the
attacker-selected conclusion. Recent black-box attacks can increase linguistic variation through
heterogeneous prompts and generators (Li et al., 2025), but such diversity remains an emergent
property of sampling rather than an explicit constraint on the retained poison set. We instead ask
whether set-level diversity can be planned, verified, and repaired directly during construction.
We introduceDnD, a targeted corpus-poisoning method built around two complementary compo-
nents. First,Divideconstructs direct-support passages with distinct document styles and contextual
support framings. Style diversity changes the register and discourse structure of each passage
while preserving the same target conclusion. This variation can disperse passage representations in
embedding space, weakening the compact clusters used by similarity-based defenses. A query-specific
style ledger discourages reuse of earlier styles and framings, while a frozen surrogate verifies that
each passage still supports the complete target relation and that the resulting set remains consistent.
Pairwise lexical overlap control further limits repeated wording. Divide therefore diversifies both the
content and expression of the poison set without sacrificing target support.
Second, direct support for the malicious answer may still compete with clean evidence in the
retrieved context. TheDoubtcomponent assigns one passage to question the evidential basis
supporting the reference answer before presenting the attacker-selected answer as the result of a later
or independent cross-check. The passage neither names the reference answer nor explicitly instructs
the model which source to trust. Together, the direct-support and Doubt passages form a coordinated
evidence set: the former establish the target claim, while the latter provides a conflict-resolution
route when competing evidence is also retrieved.
Retrieval adaptation is applied only after the passage content has been constructed and verified.
Under black-box access, each passage receives a distinct question-facing variant; under white-box
access, a separate HotFlip prefix is optimized for each passage. This separation preserves the stylistic
and contextual differences established during construction. We evaluateDnDon HotpotQA and
Natural Questions using Llama, Qwen, and Mistral across nine RAG settings: an undefended Vanilla
pipeline and eight defenses. Across both datasets and retriever-access settings,DnDmatches or
outperforms prior attacks in most configurations, with its largest gains against clustering- and
conflict-aware defenses. Ablations isolate the effects of support diversification and the Doubt passage,
while experiments with alternative dense retrievers assess transfer beyond the default retriever.
Our contributions are threefold:
•We identify correlated stylistic, semantic, and lexical patterns as a central weakness of inde-
pendently constructed multi-passage poisoning attacks against defenses that jointly analyze
retrieved evidence.
•We developDnD, which explicitly plans stylistically diverse support passages, controls pairwise
lexical overlap, and adds a complementary Doubt passage for resolving competing evidence.
•Across two open-domain QA datasets, three target LLMs, and nine RAG settings,DnD
matches or outperforms prior attacks in most configurations, with the largest gains against
clustering- and conflict-aware defenses.
2 Related Work
CorpuspoisoningattacksonRAG.Retriever-focusedattacksoptimizeinjectedpassagesforselected
queries. Zhong et al. (2023) adapt HotFlip-style gradient-guided token replacement (Ebrahimi et al.,
2

2018)topassageretrieval. End-to-endattacksadditionallytargetgeneration. PoisonedRAGseparates
retrieval and target-answer conditions but constructs passages independently for each question–target
pair (Zou et al., 2025). LIAR learns transferable adversarial content through bi-level optimization
over surrogate retrievers and generators in a query-agnostic setting (Tan et al., 2024). TPARAG
and Joint-GCG couple retrieval and generation more directly, but respectively optimize a different
attack objective and use target-generator gradients (Li et al., 2026; Wang et al., 2026). Recent
single-document attacks pursue dominance through self-contained evidence chains or attacker-chosen
false evidence (Chang et al., 2025; Zhang et al., 2026a), while RIPRAG learns from repeated
target-system feedback (Xi et al., 2026). CPA-RAG increases linguistic variation through prompt
templates and heterogeneous generators, but filters candidates individually rather than constraining
pairwise diversity in the retained set (Li et al., 2025). In contrast,DnDreceives no target-generator
outputs or gradients and no defense feedback; white-box access adds only retriever gradients. It
treats diversity as a retained-set property by coordinating document style, support framing, lexical
overlap, and a complementary Doubt role across aB-passage bundle.
Defenses against corrupted retrieval.Existing defenses intervene through post-retrieval filtering,
conflict-aware reasoning, or robust aggregation. TrustRAG combines embedding clustering with
LLM self-assessment; RAGuard filters passages using chunk-wise perplexity and unusually high
query–passage similarity; and RAGDefender combines clustering- or concentration-based grouping
with pairwise-similarity analysis (Zhou et al., 2025; Cheng et al., 2025; Kim et al., 2025). Other
methods address noisy or conflicting evidence. InstructRAG learns explicit denoising from self-
synthesized rationales, SeCon-RAG combines semantic and cluster-based filtering with conflict-aware
consistency checks, and Astute RAG consolidates retrieved evidence with elicited model-internal
knowledge (Wei et al., 2025; Si et al., 2025; Wang et al., 2025). RobustRAG isolates retrieved
evidence and securely aggregates group-level outputs, whereas ReliabilityRAG selects reliability-
weighted consistent evidence through a contradiction graph; both provide robustness guarantees
under bounded-corruption assumptions (Xiang et al., 2024; Shen et al., 2025). Across these defense
families, we evaluate whether poison sets coordinated over distinct document styles, target-support
framings, lexical realizations, and evidential roles remain effective.
3 Problem Formulation
We consider a retrieval-augmented generation (RAG) system with a clean corpus C, a retriever RK
that returns the top- Kpassages, an optional post-retrieval defense D, and a target LLM G. Let
FD,G(q,S)denote the complete downstream answer procedure applied to question qand retrieved
passage set S. This notation covers defenses that filter, rerank, or group passages before generation,
as well as defenses that aggregate outputs obtained from different passage groups. Without a defense,
FId,G(q,S) =G(q,S). For each question q, we denote its reference answer by yq. The retrieval depth
Kcontrols how many passages enter the downstream RAG pipeline, whereas the poison budget
Blimits how many passages the attacker may insert for each question. These two parameters are
independent.
3.1 Targeted Corpus Poisoning
For an evaluation set Q, the attacker preselects an incorrect target answer y⋆
q̸=yqfor each question
q∈Qand constructs a query-specific set of poisoned passages PB(q), subject to |PB(q)| ≤B.
Because all query-specific poison sets are inserted into a single shared corpus, we define the complete
3

Figure 1: Overview ofDnD. Divide constructs diverse direct-support passages, while Doubt reframes
evidence for the reference answer. The assembled set is verified and selectively repaired before each
passage receives access-specific retrieval adaptation and is inserted into the corpus.
injected set as
P=[
q′∈QPB(q′).(1)
The poisoned RAG output for questionqis then
byP(q) =F D,G(q, R K(q,C ∪ P)).(2)
The attacker modifies only the corpus: the user question, deployed retriever, defense, and target
LLM remain unchanged. Although the poisoned corpus is shared across Q, the poison budget is
enforced separately for each question through|P B(q)| ≤B.
Attack objectiveLet M(a, s)∈ {0,1}denote a fixed answer-matching function that returns1if the
normalized answer aoccurs in the normalized response s, and0otherwise. Define the per-question
strict-success indicator as Sq=M(y⋆
q,byP(q))[1−M(yq,byP(q))]. Given an evaluation set Q, the
attacker maximizes the strict targeted attack success rate (ASR):
maximize
{PB(q)}q∈Q1
|Q|X
q∈QSq
subject to|P B(q)| ≤B,∀q∈ Q.(3)
3.2 Threat Model
Attacker goal.For each question q, the attacker seeks a final RAG response containing y⋆
qbut not
yq; responses containing both are failures. The attack modifies only the corpus and must remain
effective after retrieval and any post-retrieval defense.
4

Attacker knowledge and access.The attacker knows q,yq, and the preselected y⋆
q, and may use
auxiliary language models and public embedding models. The target answer is fixed before candidate
generation. Black-box access exposes no deployed-retriever rankings, scores, parameters, or gradients;
white-box access adds only retriever gradients for retrieval-facing optimization. In both settings,
the poison set is fixed before evaluation, and construction or selection receives no retrieved clean
passages, defense decisions, target-generator outputs or logits, or observed ASR.
4 Method
4.1 Overview
Given a question q, its reference answer yq, a preselected target answer y⋆
q, and a poison budget B,
the attacker constructs a set of passages intended to steer the target LLM toward y⋆
q. A common way
to scale targeted corpus poisoning is to generate or paraphrase each poison passage independently.
Although this may vary their surface forms, it does not coordinate the passages as a set: multiple
passages can still reuse similar document styles, support framings, or salient lexical patterns. This
leaves set-level redundancy uncontrolled, particularly when a defense jointly assesses the retrieved
evidence. To address this limitation, we proposeDnD, a targeted multi-passage corpus-poisoning
method that coordinates both the realization and the evidential roles of the injected passages.
Figure 1 summarizes the two complementary components ofDnD.Divideconstructs a set of
passages that directly support y⋆
qwhile coordinating three aspects across the set: the document
style assigned to each passage, the contextual framing used to support y⋆
q, and its lexical realization.
Each passage is planned relative to the previously accepted passages so that it contributes a distinct
support route rather than an uncoordinated restatement. At the same time, every direct-support
passage must independently preserve a recoverable path toy⋆
q.
Doubtcomplements the direct-support set with a distinct evidential role. Diverse direct support
addresses only the target side of the retrieved evidence. When poison passages are retrieved alongside
clean evidence favoring yq, another positive assertion of y⋆
qmay not address why the competing
inference should be treated cautiously. Doubt therefore qualifies a plausible evidential route toward yq
and presents a target-consistent cross-check favoring y⋆
q. Consequently, one poison passage addresses
the potential answer conflict instead of repeating another direct rationale for the target answer.
In the default multi-passage setting, Divide constructs Pdir(q) ={pdir
q,i}B−1
i=1, and Doubt constructs
one complementary passagepdoubt
q. The complete poison set is
P(q) =Pdir(q)∪n
pdoubt
qo
.(4)
When B= 1,DnDomits Doubt and constructs a single direct-support passage; results for this
single-passage setting are reported in Appendix A.3.
Passage construction is separated from access-specific retrieval adaptation. The construction-time
writer proposes role-conditioned support plans and corresponding candidates, a frozen construction-
time verifier checks plan fidelity and target-answer recoverability, and a deterministic set controller
enforces the prescribed set-level overlap constraints. Failed checks trigger local refinement. Retrieval
adaptation is applied to each passage only after the complete set passes construction-time validation.
Neither construction nor validation receives feedback from the evaluated defense, the target generator,
or observed attack success.
5

4.2 Divide: Diversifying Target Support
Passage-leveleffectivenessalonedoesnotensurethatadditionalpoisonslotscontributecomplementary
evidence. Divide therefore makes direct-passage construction history-aware: each passage is planned
relative to the partial set while the target conclusion remains fixed.
Sequential style planning.Divide sequentially constructs the direct-support set Pdir(q)defined in
Equation 4. Before generating pdir
q,i, the writer creates an internal style-and-support plan ci= (si, ri).
Here, sispecifies a generated document style or discourse register, while rispecifies the target-support
framing that leads toy⋆
q.
At step i, a ledger L<i={(sj, rj)}j<isummarizes the styles and support framings assigned to
the preceding direct passages. Conditioned on q,y⋆
q, andL<i, the writer proposes a style–framing
pair not already represented in the ledger and realizes the plan as a self-contained passage supporting
the complete target conclusion. The accepted plan is then added to the ledger before the next
passage is constructed.
Consequently, later passages are generated relative to the evidence roles already present in the
partial set rather than from the same unchanged context. The same history-aware construction rule
determines each additional passage. The support plans and ledger are used only during construction
and are not inserted into the corpus.
Diversity control.Sequential planning specifies distinct styles and support framings, but the
generated passages may still drift from their assigned plans or converge on similar wording. Divide
therefore checks style fidelity and lexical overlap separately. The frozen verifier checks whether
each candidate realizes its assigned style and support framing and whether either substantially
duplicates an accepted plan already represented in the ledger. By changing the surrounding context
and discourse structure while preserving the target claim, style diversification can spread passage
representations in embedding space and weaken the compact clusters used by K-means-style defenses.
At the lexical level, a deterministic controller measures pairwise lexical overlap over the current
passage set using ROUGE-L F1 (Lin, 2004). The set must satisfy constraints on both its maximum
and mean pairwise scores. The maximum criterion captures an isolated near-duplicate pair that
could be obscured by averaging, whereas the mean criterion limits aggregate reuse across the set.
The overlap measure, acceptance thresholds, and associated implementation details are provided in
Appendix B
Verification and Refinement.Each direct-support passage is evaluated in isolation by the frozen
verifier and must make y⋆
qrecoverable from its own content. This prevents an individually weak
passage from being retained solely because other passages supply the missing support. A candidate
must therefore satisfy its assigned support plan, the target-recoverability requirement, and the
current set-level redundancy constraints.
When a check fails, the controller revises only the implicated passage while leaving the remaining
candidates unchanged. All checks are then rerun over the updated set because a local revision can
alter both target support and set-level redundancy. The direct-support set proceeds to final assembly
only after every passage passes the construction-time checks.
4.3 Doubt: Reframing Answer Conflict
Diversity among direct-support passages does not by itself address clean evidence that may favor yq.
Under multi-passage injection, Doubt assigns one passage a different evidential function: qualifying
6

a plausible competing inference and presenting a resolution that remains consistent with y⋆
q. This
role prevents all injected passages from providing only repeated positive support.
Answer-conditioned reframing.Conditioned on q,yq, and y⋆
q, the writer constructs a self-contained
passage with two linked components. First, the passage introduces a question-specific limitation in a
plausible evidential route toward yq, such as restricted scope, uncertain provenance, or ambiguous
interpretation. Second, it presents a later or independent cross-check whose stated conclusion
supports y⋆
q. The resulting passage therefore provides a target-consistent interpretation of a potential
answer conflict rather than adding another direct rationale.
The reference answer guides the choice of evidential caveat but is not explicitly stated as the
competing answer. The passage also does not directly declare yqfalse, claim access to a particular
retrieved document, or instruct the target LLM which source to trust or ignore. These constraints
avoid leaking the reference answer under the strict attack objective and preserve an evidential rather
than instructional passage form. The Doubt passage must make y⋆
qrecoverable in isolation and
satisfy the same construction-time validation requirements.
Final set assembly.After all assigned roles have been instantiated, the final poison set P(q)
combines the direct-support passages with the Doubt passage, as defined in Equation 4.
The controller reruns the redundancy checks over the complete set, and the frozen verifier
jointly probes the assembled passages for target consistency and recoverability. Any failed check
triggers local revision of the implicated passage followed by full-set revalidation. Once the final set
passes these checks, its semantic content is fixed and each passage proceeds to the access-specific
retrieval-adaptation stage.
5 Experimental Evaluation
We evaluateDnDthrough matched end-to-end comparisons and controlled component ablations.
5.1 Experimental Setup
Knowledge bases.We evaluate on HotpotQA (Yang et al., 2018) and Natural Questions (NQ)
(Kwiatkowski et al., 2019). Following the PoisonedRAG setup (Zou et al., 2025), we use the complete
Wikipedia passage corpus associated with each benchmark as its clean knowledge base, containing
5,233,329 passages for HotpotQA and 2,681,468 passages for NQ. All attacks use the same clean
corpora and indices, with query-specific poison passages appended without modifying or removing
clean passages.
Attack baselines.We reproduce PoisonedRAG (PR) (Zou et al., 2025) from its official imple-
mentation as our primary matched baseline. For broader comparison, we include CorpusPoisoning
(Zhong et al., 2023), the PromptInjection and Disinformation baselines defined by Zou et al. (2025),
Paradox (Choi et al., 2025), Adversarial Decoding (Zhang et al., 2026b), and CPA-RAG (Li et al.,
2025). They cover retrieval optimization, explicit instruction injection, unadapted false evidence,
retriever-preference-guided generation, objective-scored decoding, and multi-model retriever-aware
generation.
PR andDnDare evaluated under matched conditions. Both use the same gpt-5-mini
construction-time writer and share the frozen question–target pairs, clean corpus and corresponding
base index, poison budget B, retrieval depth K, retriever, downstream RAG systems, and evaluator,
7

while retaining their method-specific prompts and construction procedures. Under black-box (BB)
access, neither attack receives rankings,scores, parameters, or gradients from the evaluated retriever.
Under white-box (WB) access, both receive the same gradient access to Contriever (Izacard et al.,
2022) and use HotFlip with the same optimization budget.
Defense methods.We evaluate an undefended Vanilla pipeline and eight defenses: InstructRAG
(Wei et al., 2025), TrustRAG (Zhou et al., 2025), SeCon-RAG (Si et al., 2025), RAGuard (Cheng et al.,
2025), RobustRAG-Keyword (Xiang et al., 2024), Astute-RAG (Wang et al., 2025), ReliabilityRAG-
MIS (Shen et al., 2025), and RAGDefender (Kim et al., 2025). We use the same defense configuration
for every attack.
Metrics.Our primary metric isstrict-ASR(Eq. 3): success requires the final response to contain
y⋆and exclude y. Poison retrieval recall is the fraction of query-specific injected passages appearing
in the raw top- Kresults before post-retrieval defense. The overlap-control ablations report post-filter
poison-slot survival, the fraction of retrieved poison slots retained by the filter. We additionally report
target-answer precision andF 1as auxiliary metrics of target-answer realization (Appendix A.4).
DnDsettings and protocol.Unless otherwise stated, B= 5: Divide constructs four direct-support
passages and Doubt constructs one complementary passage. All construction-time settings are fixed
across conditions. For each attack–dataset–access setting, all query-specific poison passages are
inserted into one corpus; the corpus and index are then frozen and reused across target LLMs and
RAG configurations, with no regeneration based on target outputs or defense decisions. Full details
are provided in Appendix B.
5.2 Overall Attack Effectiveness
Table 1 reports the complete HotpotQA attack–generator–configuration matrix at B= 5and K= 5;
the corresponding NQ results are reported in the appendix. We focus on TrustRAG and SeCon-RAG
because both apply clustering-based filters to suppress redundant retrieved evidence.
Averaged uniformly over the 27 target-generator–configuration combinations,DnDincreases
ASR over PR from53 .37%to67 .44%under BB, a gain of14 .07percentage points, and from52 .26%
to68 .41%under WB, a gain of16 .15points. Relative to PR,DnDis higher in 19 BB cells, tied
in six, and lower in two; the corresponding WB counts are 21, four, and two. After averaging over
the nine configurations, the improvement remains positive for each target LLM under both access
settings.
Effectiveness under clustering-based filtering.Averaged over the three target LLMs, PR achieves
2.33%ASR against TrustRAG under both BB and WB, whereasDnDreaches29 .00%under BB
and36 .33%under WB. The corresponding gains are26 .67and34 .00percentage points. Against
SeCon-RAG,DnDraises ASR from2 .33%to26 .33%under BB and from2 .67%to38 .00%under
WB, yielding gains of24.00and35.33points.
DnDoutperforms PR in all 12 matched comparisons spanning three target LLMs, two clustering-
based defenses, and both access settings. The cell-level gains range from18to36points under BB
and from23to45points under WB. Moreover, amongDnD-BB, PR-BB, and the six reference
attacks,DnD-BB is uniquely highest in all six TrustRAG and SeCon-RAG cells.
These results show thatDnDremains substantially more effective than PR under clustering-
based filtering. As a separate output-level diagnostic, Appendix A.5 shows that the finalDnD
8

Target LLM AttackRAG configuration
Vanilla Instruct TrustRAG SeCon RAGuard RobustRAG Astute ReliabilityRAG RAGDefender
LlamaCorpusPoisoning 2.0 3.0 5.0 4.0 1.0 5.0 5.0 1.0 0.0
PromptInjection 49.0 25.0 5.0 10.0 4.0 24.0 27.0 39.0 3.0
Paradox 85.0 79.0 6.0 5.0 15.0 60.0 45.0 82.0 82.0
Disinformation 88.0 77.0 5.0 5.0 11.0 61.0 44.0 84.0 82.0
Adversarial Decoding 87.0 74.0 5.0 5.0 45.0 52.0 52.0 79.0 71.0
CPA-RAG 89.0 75.0 5.0 5.0 19.0 58.0 38.0 90.0 81.0
PR-BB 90.077.0 3.0 3.0 5.0 59.0 38.0 91.093.0
DnD-BB (Ours) 90.0 79.0 39.0 29.0 64.0 61.0 66.0 93.0 93.0
PR-WB 90.075.0 3.0 4.0 6.0 56.0 41.0 89.088.0
DnD-WB (Ours) 90.0 83.0 41.0 42.0 64.0 57.0 50.0 94.0 88.0
QwenCorpusPoisoning 8.0 7.0 4.0 2.0 8.0 9.0 4.0 2.0 1.0
PromptInjection 59.0 35.0 6.0 4.0 11.0 29.0 28.0 38.0 1.0
Paradox 87.0 81.0 3.0 3.0 21.0 68.0 46.0 90.0 90.0
Disinformation 90.0 78.0 7.0 4.0 16.0 68.0 47.0 92.0 93.0
Adversarial Decoding 90.0 76.0 3.0 3.0 55.0 68.0 46.0 92.0 85.0
CPA-RAG 91.0 77.0 3.0 3.0 30.0 72.0 49.0 97.0 96.0
PR-BB 91.0 76.0 2.0 2.0 8.0 74.0 56.096.0 96.0
DnD-BB (Ours) 92.0 81.0 28.0 30.0 70.0 76.0 62.093.0 93.0
PR-WB 91.078.0 2.0 2.0 10.0 72.0 49.0 92.0 87.0
DnD-WB (Ours) 91.0 80.0 42.0 47.0 73.0 75.0 51.0 93.0 88.0
MistralCorpusPoisoning 3.0 2.0 11.0 8.0 3.0 5.0 5.0 2.0 2.0
PromptInjection 41.0 23.0 12.0 14.0 5.0 31.0 22.0 39.0 0.0
Paradox 85.0 82.0 9.0 9.0 13.0 65.0 45.0 83.0 84.0
Disinformation 85.0 81.0 9.0 9.0 11.0 65.0 46.0 88.0 84.0
Adversarial Decoding 87.0 78.0 9.0 9.0 45.0 65.0 43.0 86.0 81.0
CPA-RAG 89.0 84.0 10.0 9.0 18.0 67.0 42.0 95.0 86.0
PR-BB 89.084.02.0 2.0 4.072.050.091.0 87.0
DnD-BB (Ours) 90.0 84.0 20.0 20.0 57.0 72.0 61.0 91.0 87.0
PR-WB 90.082.02.0 2.0 9.070.046.0 88.087.0
DnD-WB (Ours) 92.081.026.0 25.0 68.069.061.0 89.0 87.0
Table 1: Attack performance across nine RAG configurations (Vanilla plus eight defenses) on
HotpotQA with poison budget B= 5and retrieval depth K= 5, measured bystrict-ASR(%;
higher is better for the attacker).PRandDnDdenote PoisonedRAG and our method;BB
andWBdenote their black-box and white-box access settings. The remaining rows are six cross-
family reference baselines. RobustRAG, Astute, and ReliabilityRAG denote RobustRAG-Keyword
(KeywordAgg), Astute-RAG, and ReliabilityRAG-MIS, respectively. Within each matched PR–DnD
pair and access setting, bold marks the higher value; exact ties are bolded for both.
bundles have lower mean and maximum within-bundle Contriever cosine similarity than matched
PR bundles across all four dataset–access conditions, including BB without construction-time access
to Contriever. Section 5.3 examines how overlap control relates to post-filter poison-slot survival.
Results on the remaining configurations.Averaged over the three target LLMs,DnDimproves
ASRagainstRAGuardoverPRby58 .00pointsunderBBand60 .00pointsunderWB.AgainstAstute-
RAG, the corresponding gains are15 .00and8 .67points. Differences under Vanilla, InstructRAG,
RobustRAG-Keyword, ReliabilityRAG-MIS, and RAGDefender are smaller on average.
AcrossDnD-BB, PR-BB, and the six cross-family reference attacks,DnD-BB is uniquely
highest in 16 cells and tied for highest in eight, covering 24 of the 27 target-generator–configuration
combinations. Its average ASR of67 .44%exceeds that of the strongest reference attack, Adversarial
Decoding at55.22%, by12.22points.
9

(a) Doubt allocation (b) ROUGE-L constraints (c) Metric sensitivity
Bw/o Doubt Full∆Constraints Van. Def. Surv. Metric Van. Def. Surv.
2 45.552.8 +7.2 Max + Mean93.0 43.690.5 ROUGE-L93.043.6 90.5
3 51.758.8 +7.1Max only 93.043.890.3 Token Jaccard94.538.0 64.9
4 53.960.4 +6.4Mean only 93.2 40.9 81.0 Char. TF–IDF 93.2 36.9 73.5
5 57.464.7 +7.3Neither94.826.5 42.9 Sup. SimCSE 93.2 40.8 84.9
Scope.Panel (a) macro-averages over nine RAG configurations, two datasets, three target LLMs, and both access settings; Full
uses one Doubt passage and B−1direct-support passages, whereasw/o Doubtuses Bdirect-support passages. Panels (b)–(c)
use Qwen2.5-7B on HotpotQA and NQ under both access settings. Van. denotes Vanilla ASR; Def. averages TrustRAG and
SeCon-RAG. Surv. denotes the macro-averaged percentage of retrieved poison slots retained after the k-means/ n-gram pre-filter
used in our TrustRAG and SeCon-RAG implementations. Results are compared within panels.
Table 2: Construction ablations and overlap-metric sensitivity. ASR and survival are percentages;
∆is in percentage points and is computed before rounding. Panel (a) evaluates the allocation of
one poison slot to Doubt. Panel (b) ablates the maximum and mean ROUGE-L constraints. Panel
(c) compares overlap metrics on shared ROUGE-L-guided candidate and repair trajectories while
retaining both constraints. Higher is better for the attacker.
5.3 Ablation Studies
Table 2 reports independently reconstructed ablations of Doubt allocation and the pairwise ROUGE-L
constraints, together with fixed-trajectory metric sensitivity.
Contribution of Doubt.Across B∈ { 2,3,4,5}, Full—one Doubt passage and B−1direct-support
passages—outperforms the all-direct-support allocation at every budget by6 .4–7.3percentage points
(Table 2(a)). Because each arm independently reconstructs the complete poison set, this comparison
evaluates the intended set-level budget-allocation policy, including the interaction between Doubt
and direct support.
Pairwise lexical-overlap control.Removing both ROUGE-L constraints reduces defended ASR
from43 .6%to26 .5%and post-filter survival from90 .5%to42 .9%. By contrast, Vanilla ASR remains
between93 .0%and94 .8%across all four configurations. This separation links the benefit of overlap
control to poison-slot survival under filtering rather than unfiltered attack strength.
The two constraints contribute asymmetrically. At the point-estimate level, Max-only nearly
matches Max+Mean in defended ASR (43 .8%vs.43 .6%) and survival (90 .3%vs.90 .5%). Mean-only
remains substantially stronger than Neither in defended ASR (40 .9%vs.26 .5%) and survival (81 .0%
vs.42 .9%). Thus, the maximum constraint is the stronger individual control, while the mean
constraint remains beneficial on its own.
Sensitivity to the overlap metric.Within this fixed-trajectory comparison, ROUGE-L yields the
highest defended ASR and post-filter survival, at43 .6%and90 .5%, respectively. Token Jaccard,
character TF–IDF, and supervised SimCSE (Gao et al., 2021) yield defended ASR values of38 .0%,
36.9%, and40 .8%, with survival rates of64 .9%,73 .5%, and84 .9%. Vanilla ASR remains high for
every metric (93 .0%–94 .5%); the metric-dependent performance separation is therefore concentrated
in the filtering.
10

6 Conclusion
In this paper, we introduceDnD, a distributed corpus-poisoning attack designed for defenses that
jointly compare retrieved evidence.DnDconstructs passages with distinct document styles and
target-support framings, controls pairwise lexical overlap, and reserves one passage to cast doubt
on evidence for the reference answer. Together, these passages preserve agreement on the attacker-
chosen conclusion while dispersing their representations and wording, weakening similarity- and
conflict-aware defenses. Evaluations on HotpotQA and Natural Questions with three target LLMs
and nine RAG settings show thatDnDmatches or outperforms prior attacks in most configurations.
Our results indicate that filtering for semantic concentration alone is insufficient and motivate
defenses that can identify coordinated evidence manipulation across stylistically distinct passages.
Limitations
Our evaluation is limited in scale by available computational and API budgets. We conduct
experiments on fixed 100-question subsets of HotpotQA and Natural Questions and evaluate three
open-weight target generators; proprietary closed-source models are not included. Although we
cover a representative set of RAG defenses, the rapidly evolving defense landscape means that
some recently proposed defenses are not evaluated. Future work could extend the evaluation to
larger query sets, additional datasets, closed-source models, and newer defenses to further assess the
generalizability of our findings.
Ethical Considerations
This work studies a dual-use security problem. The proposed techniques could be misused to
manipulate answers produced by RAG systems whose knowledge bases accept untrusted content.
Our purpose is to evaluate the robustness of existing defenses under controlled conditions and to
motivate stronger protections against coordinated corpus poisoning.
All experiments were conducted offline using public research benchmarks and locally instantiated
RAG pipelines. We did not attack deployed services, modify third-party corpora, publish poisoned
passages to the Web, recruit human participants, or collect new personal data. The generated
passages were used only within the benchmark evaluation environment.
References
Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang, Yuekai Huang, Ziyou Jiang, Yang Liu,
and Qing Wang. 2025. One shot dominance: Knowledge poisoning attack on retrieval-augmented
generation systems. InFindings of the Association for Computational Linguistics: EMNLP 2025,
pages 18811–18825, Suzhou, China. Association for Computational Linguistics.
Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, and Minghong
Fang. 2025. Secure retrieval-augmented generation against poisoning attacks. In2025 IEEE
International Conference on Big Data (BigData), pages 1799–1806. IEEE.
Chanwoo Choi, Jinsoo Kim, Sukmin Cho, Soyeong Jeong, and Buru Chang. 2025. The RAG paradox:
A black-box attack exploiting unintentional vulnerabilities in retrieval-augmented generation
systems. InFindings of the Association for Computational Linguistics: EMNLP 2025, pages
23723–23744, Suzhou, China. Association for Computational Linguistics.
11

Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. 2018. HotFlip: White-box adversarial
examples for text classification. InProceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers), pages 31–36, Melbourne, Australia.
Association for Computational Linguistics.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of
sentence embeddings. InProceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association
for Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive
learning.Transactions on Machine Learning Research.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.
Minseok Kim, Hankook Lee, and Hyungjoon Koo. 2025. Rescuing the unpoisoned: Efficient defense
against knowledge corruption attacks on RAG systems. InAnnual Computer Security Applications
Conference (ACSAC).
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. 2019. Natural questions: A benchmark for question answering research.Transactions of
the Association for Computational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, volume 33, pages 9459–9474. Curran Associates, Inc.
Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, and Jianfeng Ma. 2025. CPA-RAG:
Covert poisoning attacks on retrieval-augmented generation in large language models.arXiv
preprint arXiv:2505.19864.
Zizhong Li, Haopeng Zhang, and Jiawei Zhang. 2026. Token-level precise attack on RAG: Searching
for the best alternatives to mislead generation. InFindings of the Association for Computational
Linguistics: EACL 2026, pages 3193–3206, Rabat, Morocco. Association for Computational
Linguistics.
Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. InText Summa-
rization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.
Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang, Prateek Mittal, and Aleksandra Korolova. 2025.
ReliabilityRAG: Effective and provably robust defense for RAG-based web search. InAdvances in
Neural Information Processing Systems, volume 38.
12

Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie
Duan, Yang Liu, and Xiaojun Jia. 2025. SeCon-RAG: A two-stage semantic filtering and conflict-
free framework for trustworthy RAG. InAdvances in Neural Information Processing Systems,
volume 38.
Zhen Tan, Chengshuai Zhao, Raha Moraffah, Yifan Li, Song Wang, Jundong Li, Tianlong Chen,
and Huan Liu. 2024. Glue pizza and eat rocks - exploiting vulnerabilities in retrieval-augmented
generative models. InProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, pages 1610–1626, Miami, Florida, USA. Association for Computational
Linguistics.
Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025. Astute RAG:
Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models.
InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 30553–30571, Vienna, Austria. Association for Computational
Linguistics.
Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, and Qing
Wang. 2026. Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented
Generation Systems.Proceedings of the AAAI Conference on Artificial Intelligence, 40(42):35793–
35801.
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2025. InstructRAG: Instructing retrieval-augmented
generation via self-synthesized rationales. InThe Thirteenth International Conference on Learning
Representations.
Meng Xi, Sihan Lv, Yechen Jin, Guanjie Cheng, Naibo Wang, Ying Li, and Jianwei Yin. 2026.
RIPRAG: Hack a black-box retrieval-augmented generation question-answering system with
reinforcement learning. InFindings of the Association for Computational Linguistics: ACL
2026, pages 16882–16902, San Diego, California, United States. Association for Computational
Linguistics.
Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, and Prateek Mittal. 2024.
Certifiably robust RAG against retrieval corruption. InICML 2024 Workshop on Next Generation
of AI Safety.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and
Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text
retrieval. InInternational Conference on Learning Representations.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.
Baolei Zhang, Yuxi Chen, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu, and Minghong Fang. 2026a.
Practical poisoning attacks against retrieval-augmented generation. InProceedings of the 31st ACM
Symposium on Access Control Models and Technologies, pages 33–44. Association for Computing
Machinery.
13

Collin Zhang, Tingwei Zhang, and Vitaly Shmatikov. 2026b. Adversarial decoding: Generating
readable documents for adversarial objectives. InFindings of the Association for Computational
Linguistics: EACL 2026, pages 2053–2068, Rabat, Morocco. Association for Computational
Linguistics.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and Danqi Chen. 2023. Poisoning retrieval corpora
by injecting adversarial passages. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, pages 13764–13775, Singapore. Association for Computational
Linguistics.
Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed
Haddadi, and Emine Yilmaz. 2025. TrustRAG: Enhancing robustness and trustworthiness in
retrieval-augmented generation.arXiv preprint arXiv:2501.00879.
Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. PoisonedRAG: Knowledge corruption
attacks to Retrieval-Augmented generation of large language models. In34th USENIX Security
Symposium (USENIX Security 25), pages 3827–3844, Seattle, WA. USENIX Association.
A Additional Evaluation Results
This appendix extends the main HotpotQA evaluation in four directions. We first report the
complete Natural Questions (NQ) attack–generator–configuration matrix under the default setting,
then examine sensitivity to the dense retriever, evaluate poison-budget sensitivity on HotpotQA,
and finally report auxiliary target-answer precision andF 1.
Throughout this appendix,BBandWBdenote black-box and white-box retriever access,
respectively.
A.1 Results on Natural Questions
We test whether the aggregate HotpotQA advantage transfers to NQ under the same default poison
budget and retrieval depth ( B=K= 5). Table 3 reports the complete matrix for three target LLMs
and nine RAG configurations (Vanilla plus eight defenses). We use the same 100-question evaluation
size, attack definitions, retriever-access settings, andstrict-ASRmetric as in the main evaluation.
Across the 27 matched target-generator–configuration cells,DnDincreasesstrict-ASRover PR
from54 .11%to60 .63%under BB, a gain of6 .52percentage points. Under WB, the corresponding
average increases from52 .30%to65 .81%, a gain of13 .52points. Gains are computed before rounding
the displayed averages. Relative to PR,DnDis higher/tied/lower in18 /0/9cells under BB and
16/5/6cells under WB. After averaging over the nine configurations separately for each target LLM,
DnDimproves over PR for all three generators under both access settings.
The gains are especially consistent for InstructRAG, TrustRAG, SeCon-RAG, and RAGuard: for
each configuration,DnDimproves over the access-matched PR baseline in all six generator–access
comparisons. Results for the remaining configurations are mixed. Thus, NQ supports transfer of the
aggregate advantage beyond HotpotQA, not uniform cell-wise dominance.
A.2 Sensitivity to the Dense Retriever
Setup.We assess the sensitivity ofDnD-BB to the choice of dense retriever at B=K= 5. Within
each dataset, this auxiliary run holds the evaluation questions, constructed poison passages, target
14

Target LLM AttackRAG configuration
Vanilla Instruct TrustRAG SeCon RAGuard RobustRAG Astute ReliabilityRAG RAGDefender
LlamaCorpusPoisoning 0.0 0.0 2.0 2.0 0.0 2.0 0.0 0.0 0.0
PromptInjection 40.0 25.0 4.0 3.0 0.0 12.0 18.0 17.0 0.0
Paradox 39.0 32.0 5.0 4.0 33.0 20.0 7.0 34.0 31.0
Disinformation 38.0 35.0 3.0 4.0 36.0 19.0 15.0 28.0 30.0
Adversarial Decoding 72.0 69.0 3.0 3.0 64.0 43.0 30.0 80.0 60.0
CPA-RAG 87.0 74.0 4.0 3.0 70.0 46.0 19.0 82.0 67.0
PR-BB 86.068.0 3.0 3.0 53.038.020.0 88.0 75.0
DnD-BB (Ours) 85.077.0 14.0 15.0 78.032.030.0 90.0 78.0
PR-WB 95.072.0 2.0 5.0 22.040.026.093.079.0
DnD-WB (Ours) 92.084.0 19.0 20.0 82.039.036.0 93.0 88.0
QwenCorpusPoisoning 3.0 4.0 3.0 3.0 4.0 0.0 3.0 0.0 2.0
PromptInjection 65.0 41.0 11.0 8.0 4.0 19.0 22.0 21.0 4.0
Paradox 44.0 39.0 8.0 8.0 40.0 23.0 17.0 32.0 33.0
Disinformation 40.0 37.0 4.0 4.0 40.0 18.0 15.0 32.0 31.0
Adversarial Decoding 82.0 76.0 3.0 3.0 77.0 62.0 39.0 88.0 84.0
CPA-RAG 92.0 75.0 6.0 7.0 82.0 59.0 42.0 92.0 88.0
PR-BB 92.071.0 4.0 4.0 62.063.041.092.086.0
DnD-BB (Ours) 89.085.0 23.0 22.0 83.060.044.090.089.0
PR-WB 94.072.0 4.0 7.0 27.063.0 41.0 97.0 92.0
DnD-WB (Ours) 94.0 91.0 37.0 36.0 85.0 63.033.0 94.0 89.0
MistralCorpusPoisoning 3.0 4.0 1.0 1.0 3.0 5.0 2.0 1.0 3.0
PromptInjection 43.0 29.0 7.0 9.0 3.0 20.0 14.0 16.0 3.0
Paradox 42.0 43.0 1.0 1.0 39.0 18.0 11.0 33.0 35.0
Disinformation 36.0 37.0 2.0 2.0 37.0 17.0 13.0 31.0 33.0
Adversarial Decoding 85.0 81.0 3.0 2.0 78.0 50.0 33.0 89.0 83.0
CPA-RAG 91.0 87.0 3.0 3.0 77.0 48.0 27.0 87.0 87.0
PR-BB 91.088.0 5.0 4.0 60.048.036.095.0 85.0
DnD-BB (Ours) 89.089.0 18.0 19.0 82.042.039.092.0 83.0
PR-WB 96.089.0 4.0 6.0 21.050.032.0 94.089.0
DnD-WB (Ours) 94.093.0 24.0 26.0 88.0 50.0 43.0 95.0 89.0
Table 3: Attack performance across nine RAG configurations (Vanilla plus eight defenses) on Natural
Questions (NQ) with poison budget B= 5and retrieval depth K= 5, measured bystrict-ASR
(%; higher is better for the attacker).PRandDnDdenote PoisonedRAG and our method;BB
andWBdenote their black-box and white-box access settings. The remaining rows are six cross-
family reference baselines. RobustRAG, Astute, and ReliabilityRAG denote RobustRAG-Keyword
(KeywordAgg), Astute-RAG, and ReliabilityRAG-MIS, respectively; RAGuard denotes RAGuard.
Within each matched PR–DnDpair and access setting, bold marks the higher value; exact ties are
bolded for both.
LLMs, and downstream configurations fixed while replacing Contriever with DPR-single (Karpukhin
et al., 2020) or ANCE (Xiong et al., 2021). The experiment contains eight RAG configurations:
it includes Perplexity and excludes ReliabilityRAG-MIS and RAGDefender. Because this run
is separate from the main nine-configuration evaluation, its Contriever results serve only as the
within-run reference and are not intended to reproduce the main matrix.
Table 4 averagesstrict-ASRuniformly over HotpotQA, NQ, and the three target LLMs; its
mean row additionally averages over the eight configurations. For a dataset with N= 100questions,
the poison retrieval recall is100 R/(BN), where Ris the total number of query-specific poison
passages appearing in their corresponding undefended top-Kcontexts.
Results.ANCE yields essentially the same meanstrict-ASRas Contriever (59 .0%versus58 .9%),
whereas the value with DPR-single is40 .8%. DPR-single also has a substantially lower poison
retrieval recall (42 .6%, versus85 .9%for Contriever and81 .1%for ANCE). The lower ASR therefore
co-occurs with lower poison retrieval recall. This pattern is consistent with poison retrieval recall
15

Configuration Contriever DPR-single ANCE
Vanilla 85.5 61.8 84.5
InstructRAG 80.0 54.8 76.7
TrustRAG 23.7 13.0 21.5
SeCon-RAG 22.3 14.3 22.5
Perplexity 85.2 61.2 84.7
RAGuard 69.5 48.2 70.5
RobustRAG 56.5 39.2 59.2
Astute-RAG 48.7 33.7 52.3
Mean 58.9 40.8 59.0
Poison retrieval recall85.9 42.6 81.1
Table 4: Dense-retriever sensitivity ofDnD-BB at B=K= 5. Eachstrict-ASRentry (%) is
averaged over both datasets and all three target LLMs, and the mean row assigns equal weight
to the eight RAG configurations. The final row reports the poison retrieval recall (%) before any
defense, averaged over both datasets.
Contriever DPR-single ANCE
RAG configuration Llama Qwen Mistral Llama Qwen Mistral Llama Qwen Mistral
Vanilla 86 91 88 74 74 73 84 88 86
InstructRAG 79 80 83 60 66 70 73 78 78
TrustRAG 41 29 21 20 23 17 22 29 19
SeCon-RAG 34 31 17 27 25 18 24 28 20
Perplexity 85 91 87 73 73 72 84 88 86
RAGuard 61 65 55 52 49 52 61 63 59
RobustRAG-Keyword 58 76 72 44 61 56 62 76 70
Astute-RAG 60 59 64 51 42 46 72 55 57
Poison retrieval recall98.4 52.2 86.4
Table 5: Complete dense-retriever sensitivity results forDnD-BB on HotpotQA at B=K= 5,
measured bystrict-ASR(%). Each generator entry is computed over the same 100 questions. The
poison retrieval recall is computed from the undefended retrieval results and is shared across target
LLMs; it is therefore reported once per retriever.
contributing to the difference, but does not isolate that factor, because replacing the retriever
also changes passage rankings and the complete retrieved context. Table 5 reports the complete
HotpotQA breakdown.
A.3 Poison-Budget Performance on HotpotQA
We vary the poison budget from B= 1to B= 5on HotpotQA while fixing the retrieval depth at
K= 5. The single-passage setting uses one direct-support passage without Doubt, whereas the
multi-passage settings combine direct support with a Doubt passage. Figure 2 reports PR andDnD
under matched BB and WB retriever access. Each point is the average over three target LLMs and
the same nine RAG configurations used in the main HotpotQA evaluation (27 cells).
In these 27-cell averages,DnDexceeds the access-matched PR baseline at all five budgets under
both access settings. TheDnD-BB andDnD-WB averages increase monotonically from B= 1to
B= 5, whereas PR-BB decreases from B= 1to B= 2before increasing from B= 2to B= 5.
Thus, the aggregate advantage is present in the single-passage case and persists throughout the
16

Figure 2: Poison-budget performance on HotpotQA at fixed retrieval depth K= 5. The single-
passage setting uses direct support without Doubt, whereas the multi-passage settings combine
direct support with a Doubt passage. Each point reportsstrict-ASRover three target LLMs and
nine RAG configurations (27 cells). PR andDnDare evaluated under matched black-box (BB) and
white-box (WB) retriever access. Higher is better for the attacker.
evaluated multi-passage range.
A.4 Auxiliary Metrics
To provide a finer-grained evaluation of target-answer realization, we additionally report target-
answer precision and F1. We normalize the generated response and target answer y⋆by lowercasing,
removing punctuation and the English articlesa,an, andthe, and normalizing whitespace. We then
compute their multiset token overlap. Precision measures the proportion of normalized response
tokens matched byy⋆, whileF 1jointly captures precision and target-answer coverage.
Both metrics are computed separately for each response and then equally averaged over the three
target LLMs and nine RAG configurations within each dataset.
Results.Table 6 shows that aDnDvariant achieves the highest precision and F1on each benchmark.
On HotpotQA,DnD-BB ranks first on both metrics, whileDnD-WB ranks second. Relative to
access-matched PR,DnDimproves precision and F1by 6.3 and 6.5 percentage points under BB
access and by 5.8 and 5.9 points under WB access.
On NQ,DnD-WB achieves the highest precision and F1, outperforming PR-WB by 5.7 and
6.1 percentage points, respectively. These results provide additional evidence thatDnDeffectively
realizes the target answer, with particularly consistent improvements under WB access across both
benchmarks.
A.5 Representation-Space Diversity of Final Poison Bundles
We complement the lexical-overlap analysis with a representation-space diagnostic of the final poison
bundles. ROUGE-L and Contriever cosine similarity characterize complementary properties: the
former measures lexical sequence overlap, whereas the latter measures concentration in the evaluated
retriever’s learned representation space. We therefore compare the finalDnDand PR bundles
directly under the matched conditions described in Section 5.1.
17

AttackHotpotQA NQ
Prec.F 1Prec.F 1
CorpusPoisoning 3.0 4.1 1.9 2.5
Disinformation 25.7 29.7 10.4 12.4
Paradox 25.0 29.0 9.6 11.7
Adversarial Decoding 29.3 32.7 23.4 27.6
CPA-RAG 29.6 33.0 27.5 31.5
PR-BB 27.4 31.0 25.8 30.0
DnD-BB (Ours) 33.7 37.524.2 28.3
PR-WB 25.8 29.6 22.5 26.6
DnD-WB (Ours)31.6 35.5 28.2 32.7
Table 6: Auxiliary target-answer precision and F1scores (%; higher is better) under the default
poison budget B= 5and retrieval depth K= 5. Each value is averaged over the3 ×9 = 27target-
generator–RAG-configuration cells for the corresponding dataset. The BB/WB suffixes denote the
retriever-access setting for PR andDnD. Bold and underline indicate the highest and second-highest
result in each column, respectively.
Smean↓S max↓
Dataset AccessDnDPR∆[95% CI]DnDPR∆[95% CI]
HotpotQA BB0.70890.8972−0.1884 [−0.1995,−0.1774]0.76840.9351−0.1668 [−0.1775,−0.1563]
HotpotQA WB0.66100.7031−0.0421 [−0.0493,−0.0350]0.71690.7651−0.0482 [−0.0570,−0.0394]
NQ BB0.63350.8457−0.2122 [−0.2206,−0.2038]0.73490.8997−0.1647 [−0.1742,−0.1553]
NQ WB0.60750.6555−0.0480 [−0.0567,−0.0394]0.67600.7311−0.0551 [−0.0658,−0.0447]
Table 7: Within-bundle Contriever representation concentration for the exact final B= 5poison
bundles. Lower values indicate greater representation-space dispersion. Differences are computed
asDnD−PR from paired, unrounded question-level values; brackets report paired 95% bootstrap
confidence intervals.
Metric.For each question, we use the exact B= 5passages that enter the end-to-end retrieval
pipeline. BecauseeachconstructedpassagesetisfrozenandreusedacrossthethreetargetLLMs, every
unique question-level bundle is encoded once. We encode each passage with facebook/contriever ,
using average pooling and the same 512-token input limit as the retrieval pipeline. The pooled
representations are ℓ2-normalized for cosine computation. All evaluated passages fall within the
input limit, so no passage is truncated.
Letz 1, . . . , z Bdenote the normalized embeddings of the passages in one bundle. We compute
Smean=1 B
2X
i<jz⊤
izj, S max= max
i<jz⊤
izj.(5)
ForB= 5, both statistics are computed over the ten unordered passage pairs. Smeanmeasures
aggregate within-bundle concentration, while Smaxcaptures the most similar passage pair. Lower
values indicate greater representation-space dispersion.
All comparisons are paired by question ID. Confidence intervals use 20,000 paired bootstrap
samples, and two-sidedp-values use 100,000 paired sign-flip randomizations.
18

Final-bundle comparison.Table 7 shows a consistent representation-space advantage forDnD.
Across all four dataset–access conditions,DnDachieves lower SmeanandSmaxthan matched PR,
with every paired 95% confidence interval lying below zero (p <0.0001in all comparisons).
The differences are particularly pronounced under BB access. On HotpotQA and NQ, respectively,
DnDreduces Smeanby0.1884and0 .2122, and reduces Smaxby0.1668and0 .1647. Moreover, every
BB question has lower SmeanandSmaxunderDnDthan under PR. This result is obtained without
either attack receiving rankings, scores, parameters, or gradients from the evaluated Contriever
during construction.
Under WB access, where both attacks use the same Contriever access and HotFlip optimization
budget, the differences remain consistent across both datasets.DnDhas lower Smeanfor 88% of
questions on each dataset and lower Smaxfor 83% of HotpotQA questions and 89% of NQ questions.
The paired confidence intervals remain separated from zero for both statistics.
Pre-adaptation representation structure.The representation-space difference is already present in
the frozen passage bodies before access-specific retrieval adaptation. Relative to PR,DnDreduces
body-level Smeanby0.1424on HotpotQA and0 .1353on NQ. The corresponding reductions in
body-level Smaxare0 .1428and0 .1397. The difference then remains visible after both BB and
WB retrieval adaptation. Thus, the distinct representation structure is established during passage
construction and preserved through the retrieval-facing stage.
Overall, the finalDnDbundles are less concentrated than matched PR bundles in the evaluated
Contriever representation space across every dataset and retriever-access condition. These findings
provide direct representation-level evidence that the diversity ofDnDextends beyond lexical overlap
and is preserved in the passages ultimately used by the RAG pipeline.
B Implementation Details
Data Selection and Evaluation Protocol.We shuffle each benchmark using seed=12 and traverse
the resulting fixed order. We retain the first 100 non-yes/no questions for which the reference answer
yis unambiguous and a distinct, type-compatible target answer y⋆can be registered. Eligibility
is determined before attack construction and does not use retrieval results, target-model outputs,
defense outcomes, orstrict-ASR. The resulting query–target sets are then frozen and shared across
all attacks, target LLMs, and RAG configurations.
Our default evaluation uses poison budget B= 5, retrieval depth K= 5, three target LLMs, and
nine RAG configurations (Vanilla plus eight defenses). Each reported cell evaluates the corresponding
fixed 100-query set using seed=12. PoisonedRAG andDnDuse identical query–target pairs, corpora,
poison budgets, retrievers, target models, and RAG configurations.
Models.For both PR andDnD, we use gpt-5-mini as the construction-time writer and google/gem
ma-4-E4B-it as the construction-time verifier. The three target LLMs are meta-llama/Meta-Llama
-3.1-8B-Instruct ,Qwen/Qwen2.5-7B-Instruct , and mistralai/Mistral-7B-Instruct-v0.3 . The
default dense retriever is facebook/contriever . Gemma is used only to verify construction-time
constraints. It returns structured passage- and bundle-level validation results but has no access to
retrieval results, target-model outputs, defense outcomes, or ASR.
Attack Construction.At B= 5, eachDnDbundle contains four complementary direct-support
passages and oneDoubtpassage. For each query, the writer may initiate at most three construction
trajectories. Within each trajectory, we permit at most five accepted local repairs, with no more
19

than two failed passages modified during each repair. We freeze the first bundle that satisfies all
construction checks; no retrieval- or ASR-based candidate selection is performed.
Gemmausesgreedydecodingwith do_sample=False , disabledthinking, and max_new_tokens=128 .
WemeasurepassagediversityusingpairwiseROUGE-LF1from rouge-score==0.1.2 , withstemming
enabled, over all ten unordered passage pairs. For the frozen semantic bodies, the maximum and
mean pairwise scores must not exceed0 .25and0 .220, respectively. After BB-side formatting, the
corresponding limits are0 .25and0 .235. The ROUGE-L thresholds were fixed before the reported
evaluation and were not selected using retrieval results, target-LLM outputs, defense outcomes, or
ASR on either evaluation set.
Retriever Access and Optimization.Under BB access, attack construction does not use retriever
gradients. Under WB access, we apply HotFlip independently to each passage, initialized from the
original question. We use 30 iterations, 100 candidates per iteration, dot-product optimization, and
seed=12. The semantic passage bodies remain frozen throughout retrieval optimization.
Contriever uses average pooling without L2 normalization, dot-product similarity, and exact
retrieval.
Target Generation.We run the target LLMs using the lmdeploy PytorchEngine in FP16. We pass
a temperature of0 .01,max_new_tokens=500 , batch size1, and a maximum prefill length of8192. We
report the supplied generation parameters without characterizing decoding as greedy or sampling,
because this behavior depends on the LMDeploy engine configuration.
RAGConfigurations.ThenineevaluatedRAGconfigurationsareVanilla(nodefense), InstructRAG,
TrustRAG, SeCon-RAG, RobustRAG-Keyword, ReliabilityRAG-MIS, RAGDefender, RAGuard, and
Astute-RAG. We follow the official implementations and their reported settings for all defenses, and
use identical configurations for every attack.
Following these official configurations, TrustRAG and SeCon-RAG use a threshold of0 .88;
RobustRAG uses α= 0.3and β= 3; RAGDefender uses multihop mode for HotpotQA and
singlehop mode for NQ; and RAGuard retrieves top five with clean-reference α= 0.025. All
remaining parameters follow the corresponding official implementations.
Compute.The evaluation uses up to ten NVIDIA Quadro RTX 6000 GPUs with 24GB of memory
each, totaling approximately 183 evaluation GPU-hours.
20