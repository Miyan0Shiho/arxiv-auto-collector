# EviScope: Paired Counterfactual Evidence Diagnostics for Faithful and Efficient Grounded Language Models

**Authors**: Suryadeep Singh Deswal

**Published**: 2026-09-15 12:18:19

**PDF URL**: [https://arxiv.org/pdf/2609.17081v1](https://arxiv.org/pdf/2609.17081v1)

## Abstract
Grounded language-model systems are often evaluated by final answer accuracy, yet a correct answer can be unsupported, drawn from the wrong source, or produced when evidence is insufficient or contradictory. We introduce EviScope, a paired counterfactual benchmark that holds the question fixed while adding, removing, distracting, or contradicting its evidence. EviScope-v1.1 contains 40 four-condition quartets with repaired counterfactual claims and span-level support labels for automatic evaluation. Across 960 gold-blind generations from Qwen2.5-7B, Llama 3.1 8B, and Gemini 3.5 Flash, paired metrics expose model-dependent grounding behavior that answer accuracy hides. On two local open models, an explicit evidence-action gate underperforms vanilla RAG on QCS: 0.15 vs. 0.50 for Qwen and 0.10 vs. 0.375 for Llama. Gemini reaches 0.944 joint success under both prompts, yet still answers 5% of conflict cases after contradiction insertion. EviScope therefore distinguishes unsupported answering, conflict blindness, and wrong non-answer actions rather than scoring answers alone.

## Full Text


<!-- PDF content starts -->

EviScope: Paired Counterfactual Evidence Diagnostics for Faithful and
Efficient Grounded Language Models
Suryadeep Singh Deswal
Indian Institute of Technology Roorkee
Roorkee, India
suryadeep_sd@ma.iitr.ac.in
Abstract
Grounded language-model systems are often
evaluated by final answer accuracy, yet a cor-
rect answer can be unsupported, drawn from
the wrong source, or produced when evidence
is insufficient or contradictory. We introduce
EVISCOPE, a paired counterfactual benchmark
that holds the question fixed while adding, re-
moving, distracting, or contradicting its ev-
idence. EVISCOPE-V1.1 contains 40 four-
condition quartets with repaired counterfac-
tual claims and span-level support labels for
automatic evaluation. Across 960 gold-blind
generations from Qwen2.5-7B, Llama 3.1 8B,
and Gemini 3.5 Flash, paired metrics expose
model-dependent grounding behavior that an-
swer accuracy hides. On two local open models,
an explicit evidence-action gate underperforms
vanilla RAG on QCS: 0.15 vs. 0.50 for Qwen
and 0.10 vs. 0.375 for Llama. Gemini reaches
0.944 joint success under both prompts, yet still
answers 5% of conflict cases after contradiction
insertion. EVISCOPEtherefore distinguishes
unsupported answering, conflict blindness, and
wrong non-answer actions rather than scoring
answers alone.
1 Introduction
Retrieval-augmented generation (RAG) is widely
used to ground language models in external evi-
dence. But a final answer can be correct for the
wrong reason: the model may answer from para-
metric memory, cite an irrelevant passage, ignore
retrieved evidence, or answer despite missing or
conflicting evidence. These errors require differ-
ent remedies, yet answer-only evaluation often col-
lapses them into the same score.
The central idea in EVISCOPEis paired evi-
dence intervention. We hold the question fixed
and change only the evidence context. A grounded
system should answer and cite support when evi-
dence is present, remain stable when irrelevant pas-
sages are added, abstain when support is removed,and flag conflict when a contradictory source is
inserted. This tests whether a system is sensitive to
evidence rather than merely capable of producing
the answer.
Our contribution is a benchmark/evaluator pack-
age rather than a new model. EVISCOPE-V1.1
constructs 160 examples from public SQuAD ques-
tions (Rajpurkar et al., 2016): 40 base questions,
each paired with sufficient, noisy, insufficient, and
conflicting evidence variants. The evaluator re-
ports conventional grounding metrics plus quartet-
level evidence removal sensitivity, noise robust-
ness, conflict sensitivity, quartet consistency, and
cost-adjusted grounded utility. A complete lo-
cal/API model study demonstrates that an intu-
itively stronger evidence-action prompt can under-
perform a simpler prompt on both local open mod-
els, while a stronger API model reduces but does
not eliminate conflict blindness.
2 Related Work and Positioning
Citation and factuality.ALCE evaluates an-
swer generation with citations (Gao et al., 2023);
FActScore decomposes factuality into atomic
claims (Min et al., 2023); and HaluEval studies
hallucination recognition (Li et al., 2023). EVIS-
COPEfollows their evidence-centric motivation but
adds paired interventions that reveal whether the
same question is handled differently as support is
added, removed, distracted, or contradicted.
Efficient grounding.Self-RAG, RA-DIT,
Adaptive-RAG, RECOMP, and context-aware de-
coding improve retrieval use, routing, compression,
or reliance on context (Asai et al., 2024; Lin et al.,
2024; Jeong et al., 2024; Xu et al., 2024; Shi et al.,
2024). EVISCOPEtests whether such choices
preserve faithfulness while changing cost.
Reasoning and source identification.Sufficient
Context separates insufficient retrieval from model
arXiv:2609.17081v1  [cs.CL]  15 Sep 2026

Benchmark Ans. Cite Suff. Abst. Conf. Pair
ALCE✓ ✓– – – –
FActScore✓– – – – –
HaluEval – – –✓– –
Sufficient
Context✓–✓ ✓– –
BRIGHT – – – – – –
MTRAG✓– partial partial – –
GoldenViewVQA ✓ ✓– – – –
EVISCOPE✓ ✓ ✓ ✓ ✓ ✓
Table 1: Positioning of EVISCOPE. “Suff.” means ex-
plicit evidence-sufficiency labels. “Paired” means the
same question is evaluated under counterfactual evi-
dence variants.
Variant Evidence Expected
actionFailure
Suff. support included answer + cite evidence-use
failure
Noisy support +
distractorsanswer + cite distractor
sensitivity
Insuff. support removed abstain memory-based
answer
Conflict support +
contradictionflag conflict conflict
blindness
Table 2: Paired counterfactual evidence design. The
question is held fixed while the evidence context
changes.
failures (Joren et al., 2025); BRIGHT stresses
reasoning-intensive retrieval (Su et al., 2025);
MTRAG studies multi-turn RAG (Katsis et al.,
2025); and GoldenViewVQA evaluates visual
evidence-source identification (Wang et al., 2026).
Compared with broader input-perturbation evalu-
ations, EVISCOPElinks support, distractors, re-
moval, contradiction, and paired transition metrics
in one grounded-evidence quartet.
3 Benchmark Construction
EVISCOPE-V1.1 starts from 40 public SQuAD
question-answer-context records with at most three
base examples per source title for topical diversity.
For each base question we create four context vari-
ants, yielding 160 examples. Table 2 shows the
intervention design.
Gold labels include answer aliases, expected ac-
tion, answerability, supporting and conflicting doc-
ument IDs, and support spans. Version 1.1 freezes
v1 as legacy and repairs every conflict passage un-
der six constraints: it must be grammatical, directly
answer the question with a same-type alternative,
contradict the support, exclude the original answer,
modify the answer-bearing claim where possible,and contain no conflict cue. Strict lints and inter-
nal consistency checks cover all 40 quartets be-
fore external validation. Two external annotators
independently reviewed all 160 source-role- and
label-blinded examples. Inter-annotator agreement
was 0.994 for evidence status and expected behav-
ior (Cohen’s κ= 0.990 ), and 1.000 for conflict
presence ( κ= 1.000 ). Against hidden gold labels,
expected-behavior and evidence-status agreement
averaged 0.997 across annotators ( κ= 0.995 ); an-
notator A matched all gold actions, while annota-
tor B differed on one example. Both annotators
matched the gold conflict-present label on all 40
conflict rows. Agreement was lower for secondary
answer-valid and citation-valid QA notes (0.744,
κ= 0.589 ), mainly because conflict rows invited
different choices betweenunclearandno/not ap-
plicable; these fields are excluded from primary
benchmark labels and retained only as audit notes.
4 Metrics
The deterministic evaluator reports answer ac-
curacy; document-level source accuracy; span
support precision, recall, and F1; right-answer-
wrong-source, right-source-wrong-answer, and
unsupported-citation rates; abstention precision, re-
call, and F1; conflict accuracy; hallucination when
non-answer behavior is expected; and average to-
kens, calls, and latency. Answers are normalized
and matched by exact alias or alias containment;
citations are exact document-ID checks; conflict
is the structured conflict /flag_conflict action;
span matches require the same document ID and
normalized quote containment. No LLM judge is
used. Joint success requires a correct answer with
correct evidence, a correct abstention, or a correct
conflict flag.
It also reports paired metrics over each four-
variant quartet. Let Si,Ni,Ii, and Cidenote suc-
cess on sufficient, noisy, insufficient, and conflict-
ing variants for base question i. For sufficient/noisy,
success means correct answer plus correct support
source; for insufficient, it means abstention; for
conflicting, it means flagging conflict. The flagship
score is quartet consistency:
QCS =1
NX
i1[S i∧N i∧I i∧C i].
We also report evidence removal sensitivity 1[S i∧
Ii], noise robustness 1[S i∧N i], conflict sensitivity
1[S i∧C i], and grounded utility per call, defined as

Model Pol. Ans. Joint ERS NR CS QCS C-Ans.
Qwen Van. .738 .819 .750 .675 .575 .500 .200
Qwen Gate .738 .644 .150 .675 .625 .150 .100
Llama Van. .700 .775 .825 .575 .550 .375 .275
Llama Gate .850 .706 .800 .775 .175 .100 .775
Gemini Van. .912 .944 .925 .900 .875 .850 .050
Gemini Gate .912 .944 .900 .900 .875 .875 .050
Table 3: Main EVISCOPE-V1.1 results. Ans. is answer
accuracy on answerable rows; C-Ans. is conflict-blind
answering after contradiction insertion. Full token/cost
fields are exported with the reproducibility artifacts.
joint success divided by average calls. These met-
rics test evidence sensitivity directly: whether be-
havior changes appropriately when the same ques-
tion receives different evidence. For paired policy
comparisons, we use 10,000-sample paired boot-
strap intervals with seed 42 and an exact two-sided
McNemar test over quartet success.
5 Reference Systems
Our main experiment compares two one-
completion policies on three models: Qwen2.5 7B
via Ollama ( qwen2.5:7b-instruct , Q4_K_M),
Llama 3.1 8B via Ollama ( llama3.1:8b ),
and Gemini 3.5 Flash via the Gemini API
(gemini-3.5-flash ). Vanilla RAG directly an-
swers from the supplied documents with citations
and may abstain or flag conflict. The evidence-
action gate explicitly classifies the context as
answerable, insufficient, or conflicting before
selecting the same output actions; it is a diagnostic
prompt baseline, not a proposed solution. Both
policies receive only the question and document
IDs/text, use temperature zero, and return an
identical structured schema. The two prompts
differ only in whether they explicitly ask the model
to classify the evidence state before choosing the
final action; the document inputs, output schema,
temperature, and parser are otherwise identical.
The final matrix contains 960 logical generations,
160 examples per model-policy pair, with zero
unresolved parse failures.
6 Results
Table 3 shows that paired evidence metrics re-
veal differences that answer accuracy alone hides.
On Qwen2.5 7B, both prompts have 0.738 an-
swer accuracy and 0.675 noise robustness, but the
gate sharply degrades evidence removal sensitivity
(0.750 to 0.150) and QCS (0.500 to 0.150). TheModel Pol. Insuff. Conflict C-Ans.
Qwen Van. 1.00 .800 .200
Qwen Gate .200 .900 .100
Llama Van. 1.00 .700 .275
Llama Gate .900 .225 .775
Gemini Van. 1.00 .950 .050
Gemini Gate 1.00 .950 .050
Table 4: Failure-mode breakdown. “Insuff.” is correct
abstention after evidence removal; “Conflict” is correct
conflict flagging; C-Ans. is answering despite conflict-
ing evidence.
gate flags more conflicts, yet it correctly abstains
on only 8 of 40 insufficient variants; most remain-
ing cases are mislabeled as conflict rather than an-
swered.
Llama 3.1 8B adds a second open-weight be-
havior. Its gate improves answer accuracy (0.850
vs. 0.700) and noise robustness (0.775 vs. 0.575),
but collapses conflict sensitivity (0.175 vs. 0.550)
and QCS (0.100 vs. 0.375) by answering 77.5%
of conflicting variants. Thus the gate is not a reli-
able prompt-only fix; it changes the failure mode
differently across local models.
Gemini 3.5 Flash substantially improves overall
grounding: both policies reach 0.944 joint success,
0.900 noise robustness, and 0.875 conflict sensitiv-
ity. The gate has a small QCS edge over vanilla
(0.875 vs. 0.850), but the paired McNemar test is
inconclusive because only one quartet is discordant.
Both Gemini policies still answer 5% of conflict
cases, showing that stronger instruction-following
reduces but does not eliminate conflict blindness.
For Qwen, gate-minus-vanilla differences are
−0.60 ERS (bootstrap 95% CI [−0.75,−0.45] ),
0NR,+0.05 CS, and −0.35 QCS; 15 QCS-
discordant quartets favor vanilla and one favors the
gate (exact McNemar p= 0.00052 ). For Llama,
differences are −0.025 ERS, +0.20 NR,−0.375
CS, and −0.275 QCS; 13 QCS-discordant quartets
favor vanilla and two favor the gate ( p= 0.0074 ).
For Gemini, differences are −0.025 ERS, 0NR,0
CS, and +0.025 QCS, with one QCS-discordant
quartet favoring the gate ( p= 1.0 ). Thus the gate’s
effect is model-dependent rather than a generally
reliable improvement.
Efficiency.Both local models use one call per
example. Qwen averages 679/687 tokens and
3.17/3.48 seconds for vanilla/gate; Llama averages
633/645 tokens and 3.04/3.54 seconds. Gemini
required retry calls during a high-demand period,

averaging 1.81 calls for vanilla and 2.09 for the
gate; the latest logical rows have zero parse fail-
ures. Logical accuracy metrics are computed over
one retained output per example; retries affect only
cost and utility. We report call-normalized utility as
an engineering cost measure, not a model-quality
metric. Gemini used 195,442 input tokens and
32,271 output tokens across both policies, for an
estimated API cost of $0.56 under the recorded
pricing metadata.
7 Qualitative Errors
The error exporter surfaces failures that row-level
accuracy conceals. For “When did Beyonce start
becoming popular?”, Qwen vanilla answers from
support but repeats the supported claim after contra-
diction insertion. Conversely, for the Sichuan earth-
quake year, Qwen gate answers correctly with sup-
port but changes toconflictafter support removal,
although no incompatible answer remains. Llama
gate often has the opposite pathology: it answers
conflicting variants instead of flagging them; for the
Sichuan earthquake year, it answers “2008” while
a conflicting document asserts “2010.” Gemini’s
residual errors are rarer but still diagnostic: both
prompts answer 5% of conflict variants. These dis-
tinctions would disappear in a metric that merged
abstention and conflict flags.
8 Conclusion
EVISCOPEevaluates whether grounded models re-
spond appropriately as evidence changes. Across
Qwen2.5 7B, Llama 3.1 8B, and Gemini 3.5 Flash,
paired interventions show both prompt-level nega-
tive results and a stronger-model ceiling: explicit
routing can degrade QCS on local models, while
Gemini reaches high joint success but still exhibits
conflict blindness. Grounded evaluation should
report paired evidence transitions and distinguish
wrong non-answer actions, not score answers and
citations in isolation.
9 Limitations
EVISCOPE-V1.1 is compact and SQuAD-derived;
its counterfactual conflicts omit source authority,
recency, and multi-document synthesis. All con-
flicts require flagging, whereas realistic systems
may resolve some using metadata. The full-160
external validation supports the action, evidence-
status, and conflict-presence labels, but the com-
pleted sheets collect categorical answer/citation-valid judgments rather than explicit support-span or
support-document aliases; fine-grained source/span
validation remains limited. Two local open-weight
models, one proprietary API model, and two
prompts still limit model-level generalization. The
result nevertheless demonstrates why insufficient
and conflicting evidence require separate actions
and paired metrics.
Ethics and Responsible Research
The benchmark derives questions and passages
from public SQuAD data. Counterfactual edits
are labeled synthetic and are not factual claims
about the world. Internal consistency checks are
kept separate from external human validation. The
two external annotators volunteered, gave consent,
and worked independently using a fixed codebook.
Model prompts contain no gold labels, and the re-
leased artifacts contain no API credentials. This
work received no external funding. The author
declares no conflicts of interest.
Acknowledgments
OpenAI Codex was used for coding assistance and
proofreading. The author reviewed and verified all
AI-assisted outputs and remains responsible for the
final content. External human annotations were
supplied separately by human annotators.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions.
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6465–6488.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-RAG:
Learning to adapt retrieval-augmented large language
models through question complexity. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 7036–7050.
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-
Cheng Juan, Ankur Taly, and Cyrus Rashtchian. 2025.
Sufficient context: A new lens on retrieval augmented
generation systems. InInternational Conference on
Learning Representations.

Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
Shah, Huaiyu Zhu, Danish Contractor, and Marina
Danilevsky. 2025. MTRAG: A multi-turn conversa-
tional benchmark for evaluating retrieval-augmented
generation systems.Transactions of the Association
for Computational Linguistics, 13:784–808.
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun
Nie, and Ji-Rong Wen. 2023. HaluEval: A large-
scale hallucination evaluation benchmark for large
language models. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 6449–6464.
Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi,
Maria Lomeli, Rich James, Pedro Rodriguez, Jacob
Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettle-
moyer, and Scott Yih. 2024. RA-DIT: Retrieval-
augmented dual instruction tuning. InInternational
Conference on Learning Representations.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike
Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023.
FActScore: Fine-grained atomic evaluation of factual
precision in long form text generation. InProceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 12076–12100.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions
for machine comprehension of text. InProceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing, pages 2383–2392.
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia
Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024.
Trusting your evidence: Hallucinate less with context-
aware decoding. InProceedings of the 2024 Confer-
ence of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (Volume 2: Short Papers), pages
783–791.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan
Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jin-
sung Yoon, Sercan Ö. Arık, Danqi Chen, and Tao Yu.
2025. BRIGHT: A realistic and challenging bench-
mark for reasoning-intensive retrieval. InInterna-
tional Conference on Learning Representations.
Yimu Wang, Yee Man Choi, Barry Zhang, Mozhgan
Nasr Azadani, Sean Sedwards, and Krzysztof Czar-
necki. 2026. Where does the answer come from?
benchmarking view-level visual evidence identifica-
tion in multi-view MLLMs for autonomous driving.
Preprint, arXiv:2606.09644.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RE-
COMP: Improving retrieval-augmented LMs with
compression and selective augmentation. InInterna-
tional Conference on Learning Representations.

A Additional Qualitative Examples
Cond. Evidence shown (shortened) Expected
action
Suff. Support states that the
Sichuan earthquake occurred
in 2008.Answer
“2008” and
cite support.
Noisy The same support is mixed
with unrelated passages about
To Kill a Mockingbirdand
Beyonce.Answer
“2008” and
cite support.
Insuff. The support is removed; only
unrelated passages remain.Abstain.
Conflict Support states 2008, while a
counterfactual passage states
2010 for the same
earthquake.Flag conflict.
Table A1: Example quartet from EVISCOPE-V1.1.
Model/pol. Case Paired behavior
Qwen gate Evidence
removalAnswers “2008” with
support, but changes to
conflictafter support is
removed even though no
incompatible answer
remains.
Llama gate Conflict
blindnessAnswers “2008” on a
conflicting variant where
another document asserts
“2010.”
Gemini
vanillaResidual
conflict
blindnessAnswers from the support
document in a
genocide-convention
conflict case rather than
flagging the incompatible
answer claim.
Table A2: Representative paired failure cases.Human validation codebook.Two external anno-
tators with technical reading proficiency and expe-
rience reading structured QA/evidence examples
worked independently on source-role- and label-
blinded sheets. They saw only the question and
randomized document aliases/text. They were in-
structed not to use outside knowledge, model out-
puts, gold labels, or the other annotator’s file.
Evidence status Decision rule
sufficientThe provided documents directly
establish an answer to the question.
insufficientThe provided documents do not
establish an answer.
conflictingRelevant documents make
incompatible claims about the answer.
ambiguousMore than one reasonable
evidence-state judgment is possible.
Table A3: Evidence-state decision rules.
Annotation field Allowed values / instruction
Expected behavioranswer,abstain, or
flag_conflict.
Answer validyes,no, orunclear.
Citation validyes,no,not_applicable, or
unclear.
Conflict presentyes,no, orunclear.
Notes Short explanation, especially for
ambiguous or conflicting rows.
Table A4: Human validation fields.