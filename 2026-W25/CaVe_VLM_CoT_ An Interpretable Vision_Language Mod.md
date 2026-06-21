# CaVe-VLM-CoT: An Interpretable Vision-Language Model Framework

**Authors**: Sneha Rao, Shaina Raza, Dhanesh Ramachandram

**Published**: 2026-06-16 18:28:47

**PDF URL**: [https://arxiv.org/pdf/2606.18385v1](https://arxiv.org/pdf/2606.18385v1)

## Abstract
Vision-Language Models (VLMs) remain prone to hallucinations, producing fluent but visually unfaithful outputs. Existing chain-of-thought and retrieval-augmented methods only partially address this, as they neither enforce step-level citation grounding nor route verification failures back to retrieval for correction. We present CaVe-VLM-CoT, a modular reflection-based agentic-RAG framework that enforces evidence-grounded reasoning through a five-stage closed-loop pipeline: Extractor, Retriever, Solver, Citation Injector, and Verifier, in which detected ungrounded claims trigger structured feedback to the Extractor for targeted re-retrieval. Since no existing framework jointly measures retrieval quality, step-wise citation faithfulness, and cross-modal grounding, we propose a suite of 23 component-wise metrics across all stages, anchored by CaVeScore, a composite metric weighting accuracy, citation precision and recall, attribution, and evidence grounding. Without any architectural or prompt modifications, CaVe-VLM-CoT achieves 87.1\% accuracy and 56.6\% CaVeScore on ScienceQA , and 55.2\% accuracy and 35.7\% CaVeScore on MMMU (30 subjects).

## Full Text


<!-- PDF content starts -->

CaVe-VLM-CoT: An Interpretable Vision-Language Model
Framework
Sneha Rao Shaina Raza Dhanesh Ramachandram
Vector Institute, Toronto, ON M5G 0C6, Canada
{sneha.rao, shaina.raza, dhanesh.ramachandram}@vector institute.ai
Abstract
Vision-Language Models (VLMs) remain
prone to hallucinations, producing ﬂuent
but visually unfaithful outputs. Existing
chain-of-thought and retrieval-augmented
methods only partially address this, as
they neither enforce step-level citation
grounding nor route veriﬁcation fail-
ures back to retrieval for correction.
We present CaVe-VLM-CoT, a modu-
lar reﬂection-based agentic-RAG frame-
work that enforces evidence-grounded rea-
soning through a ﬁve-stage closed-loop
pipeline: Extractor, Retriever, Solver, Ci-
tation Injector, and Veriﬁer, in which de-
tected ungrounded claims trigger struc-
tured feedback to the Extractor for tar-
geted re-retrieval. Since no existing frame-
work jointly measures retrieval quality,
step-wise citation faithfulness, and cross-
modal grounding, we propose a suite
of 23 component-wise metrics across all
stages, anchored by CaVeScore, a com-
posite metric weighting accuracy, citation
precision and recall, attribution, and ev-
idence grounding. Without any archi-
tectural or prompt modiﬁcations, CaVe-
VLM-CoT achieves 87.1% accuracy and
56.6% CaVeScore on ScienceQA , and
55.2% accuracy and 35.7% CaVeScore on
MMMU (30 subjects). We release the code
for reproducibility /githubProject code
1 Introduction
VLMs take an image and a question and produce
an answer through some form of reasoning. Sys-
tems have advanced considerably from early VQA
models ( Antol et al. ,2015) to instruction-tuned
multimodal models such as LLaVA ( Liu et al. ,
2023), GPT-4V ( Achiam et al. ,2023), and In-
structBLIP ( Dai et al. ,2023). Despite so much
progress, these models frequently generate ﬂuent
but unfaithful responses, hallucinating objects, ev-
idence, and reasoning steps unsupported by the vi-
sual or textual context ( Li et al. ,2023;Bai et al. ,
2024). Reported hallucination rates exceed 40% on
knowledge-intensive queries ( Wu et al. ,2026), a se-rious concern in domains such as medicine, ﬁnance,
and education, where errors are costly.
Two research directions partially address this prob-
lem. Chain-of-thought (CoT) prompting ( Wei
et al.,2022) exposes intermediate reasoning steps,
with multimodal extensions such as LLaVA-
CoT ( Xu et al. ,2025) adding structured output
formats or post-hoc visual grounding. Retrieval-
augmented generation (RAG) ( Lewis et al. ,2020)
grounds outputs in retrieved evidence, with multi-
modal variants such as RIV-CoT ( Corbiere et al. ,
2025) and MI-RAG ( Choi et al. ,2025) retrieving
evidence during reasoning. However, these ap-
proaches leave three important gaps unaddressed:
(1) the absence of step-level citation mechanisms
that require each reasoning step to be grounded in
speciﬁc retrieved evidence; (2) the lack of an ex-
plicit veriﬁcation stage to determine whether re-
trieved context has been faithfully incorporated
into the reasoning process; and (3) the absence of
a self-correcting feedback loop capable of tracing
ungrounded claims back to the Extractor for
corrective re-querying and re-retrieval.
In this work, we present CaVe-VLM-CoT (Cite-
and-Verify Vision-Language Model with Chain-
of-Thought), a modular reﬂection-based agentic-
RAG framework that addresses all three gaps si-
multaneously. Rather than treating retrieval, rea-
soning, and veriﬁcation as separate concerns, it in-
tegrates them into a ﬁve-stage closed-loop pipeline
comprising an Extractor ,Retriever ,Solver ,
Citation Injector andVerifier , looping until
reasoning is veriﬁably grounded or a retry budget
is exhausted.
The focus of existing evaluation frameworks is
mostly accuracy-centric, for example, VQA bench-
marks score only answer accuracy, ALCE ( Gao
et al. ,2023) evaluates citation quality but is
text-only, and VLM judges such as LLaVA-
Critic ( Xiong et al. ,2025) rate responses holis-
tically without attributing failures to speciﬁc
stages. In contrast, CaVe-VLM-CoT provides ﬁne-
grained process supervision by enforcing evidence-
grounded reasoning, verifying the faithfulness of
intermediate and ﬁnal outputs, and enabling cor-
rective retrieval when unsupported claims are de-arXiv:2606.18385v1  [cs.AI]  16 Jun 2026

tected. This design not only improves answer re-
liability but also increases transparency by expos-
ing where and why failures occur within the re-
trieval–reasoning–veriﬁcation pipeline.
Our contributions are: (1) Step-level cita-
tion enforcement with a closed veriﬁca-
tion loop. To our knowledge, CaVe-VLM-CoT
is the ﬁrst evaluation framework to jointly en-
force inline citation at every reasoning step, val-
idate each citation against its source, and route
structured failure signals back to the Extrac-
torfor corrective re-querying and re-retrieval.
(2)A 23-metric evaluation suite for citation-
grounded multimodal reasoning across four
evaluation axes (Extractor, Retriever, Solver, Ver-
iﬁer), anchored by CaVeScore , a composite met-
ric that jointly weights accuracy, citation preci-
sion and recall, attribution (AIS), and evidence
grounding. (3) Cross-dataset diagnostic analy-
sis.On ScienceQA ( n=1,000), the system achieves
87.1%accuracy and 56.6%CaVeScore; on MMMU
(n=500, 30 subjects), 55.2%and35.7%respec-
tively. Stage-level analysis identiﬁes retrieval
coverage as the primary bottleneck on out-of-
domain data. This paper contributes an archi-
tectural framework and evaluation methodology
rather than a claim of state-of-the-art accuracy;
the empirical results serve as a diagnostic founda-
tion for future work.
2 Related Work
Hallucination in VLMs. VLMs generate plau-
sible but unfaithful outputs across many tasks ( Li
et al.,2023;Bai et al. ,2024); CHAIR ( Rohrbach
et al.,2018) and AMBER ( Wang et al. ,2023)
quantify object and relational hallucinations, and
Wu et al. (2026) report hallucination rates above
40% on knowledge-intensive queries even for
instruction-tuned models. The diagnostic lit-
erature attributes hallucination to weak visual
grounding, incomplete parametric knowledge, or
absent external grounding, but none enforce step-
wise grounding within a reasoning chain.
CoT for VLMs. CoT prompting ( Wei et al. ,
2022) exposes intermediate steps; LLaVA-CoT ( Xu
et al.,2025) adapts this to a four-stage multimodal
format, Zhang et al. (2025b ) distil GPT-4o traces
into smaller models and Critic-V ( Zhang et al. ,
2025a ) trains a separate VLM critic. These chains
run in a single pass with no per-step citation re-
quirement, and their veriﬁcation, where it exists,
is a passive ﬁlter that never routes a failure verdict
back into retrieval.
RAG for multimodal reasoning. RAG ( Lewis
et al.,2020) grounds generations in retrieved evi-
dence; ALCE ( Gao et al. ,2023) formalises citation-
aware generation and Ji et al. (2024) combineCoT with citation supervision. Multimodal vari-
ants RIV-CoT ( Corbiere et al. ,2025) and MI-
RAG ( Choi et al. ,2025) retrieve evidence for vi-
sual CoT but do not verify faithful use of it. Self-
RAG ( Asai et al. ,2024) add self-reﬂection, but
reﬂection and generation share one forward pass
and failure signals never reach the retrieval stage.
CaVe-VLM-CoT instead veriﬁes with a dedicated
module and propagates failure signals back to the
Extractor , surfacing fresh evidence rather than
re-reasoning over the same context.
Evaluation of citation quality. VQA bench-
marks ( Antol et al. ,2015;Lu et al. ,2022;Marino
et al.,2019) report only accuracy; ALCE ( Gao
et al.,2023), FActScore ( Min et al. ,2023), and AIS-
style metrics ( Ji et al. ,2024) evaluate citation qual-
ity in text only; and multimodal judges LLaVA-
Critic ( Xiong et al. ,2025), Prometheus-Vision ( Lee
et al.,2024a ), and VHELM ( Lee et al. ,2024b ) score
responses holistically without decomposing failures
by stage. CaVe-VLM-CoT’s 23-metric suite spans
evaluation axes (Extractor, Retriever, Solver, Ci-
tation Injector, Veriﬁer) and is, to our knowledge,
the ﬁrst to jointly evaluate retrieval quality, step-
wise citation faithfulness, and cross-modal ground-
ing within an agentic-RAG pipeline. In sum, prior
work neither enforces step-level citation, nor veri-
ﬁes it correctively, nor routes failure signals back
to retrieval; CaVe-VLM-CoT closes all three gaps.
3 Methodology
CaVe-VLM-CoT is built around a single architec-
tural commitment that every factual claim in a rea-
soning chain must be traceable to a speciﬁc piece
of retrieved evidence, and any chain that fails this
standard must trigger targeted re-retrieval rather
than being silently accepted. Figure 1provides a
system overview; full prompt templates and imple-
mentation details are in Appendices BandE.
3.1 Pipeline Architecture
CaVe-VLM-CoT is a reﬂection-based agentic
RAG pipeline ( Shinn et al. ,2023;Asai et al. ,2024)
comprising ﬁve sequentially connected modules,
Extractor ,Retriever ,Solver ,Citation In-
jector , andVerifier , with a rule-based feedback
loop from the Verifier to the Extractor , exe-
cuted for up to three attempts ( M=3).
Extractor The Extractor decomposes each
question into targeted sub-queries that should col-
lectively surface the evidence needed for a cor-
rect answer. Hint text, lecture content, and sub-
ject labels are withheld, forcing sub-queries to be
grounded in the question itself. A few-shot prompt
with four in-context examples elicits a structured
query list; imperfect outputs are handled by a
three-tier fallback parser (Appendix B.1). Af-

Figure 1: Overview of the CaVe-VLM-CoT pipeline. A multimod al question and its images enter at
left. The Extractor (unsloth/Qwen2.5-7B-Instruct-bnb-4 bit) decomposes the question into sub-queries
routed to a local Knowledge Base and live Web Search; retriev ed chunks and input images pass to the
Solver (LLaVA-CoT), which produces a structured reasoning trace. The Citation Injector ﬁlls uncited
claims via cross-encoder matching, and the Veriﬁer (Qwen2. 5-VL-32B) audits every citation. On detected
hallucinations, structured feedback returns to the Extrac tor for another retrieval attempt, looping until
reasoning is veriﬁed or the retry budget is exhausted.
ter parsing, a deterministic augmentation step ap-
pends one choice-discriminating query per answer
option (e.g., "sturgeon bottom feeding mouth
adaptation" ), guaranteeing retrieval coverage for
every candidate.
On retry attempts, the Extractor additionally re-
ceives a structured feedback record from the Ver-
iﬁer specifying the failing claim, the failure type
(fake citation ,misrepresented evidence , or
fabricated fact ), what the cited evidence actu-
ally says, and a suggested retrieval direction. Pre-
viously used queries are listed explicitly to discour-
age repetition. Figure 1illustrates this feedback
loop with a concrete example. Full prompt design
is described in Appendix B.1.
Retriever The local knowledge base indexes the
ScienceQA lecture corpus using dual FAISS ( Douze
et al.,2025) dense and BM25 ( Robertson and
Walker ,1994) sparse indices, fused via Recipro-
cal Rank Fusion ( Cormack et al. ,2009). For each
sub-query, the Retriever executes both local re-
trieval and web search, including choice-augmented
query variants for comparative questions. A cross-
encoder reranker scores all pooled candidates, with
top local chunks reserved unconditionally to guar-
antee the knowledge base coverage. All evidence is
keyed to its originating sub-query for traceability.
Corpus preprocessing and indexing details are in
Section E.Solver The Solver produces the reasoning trace
and ﬁnal answer in a structured three- or four-
block format ( Observations ,Summary ,Rea-
soning ,Conclusion ), making each stage sepa-
rable for downstream citation injection and veri-
ﬁcation. Question images are passed directly to
the Solver rather than routed through retrieval, as
both sliding-window patch and Grounding-DINO-
based ( Liu et al. ,2024) region proposals degraded
performance relative to text-only retrieval in pre-
liminary experiments.
Citation rules are speciﬁed negatively: zero cita-
tions is preferable to a fabricated one, and domain
knowledge claims must be explicitly ﬂagged. A
cross-encoder consistency check corrects the stated
answer letter when the reasoning text scores more
strongly for a diﬀerent choice. Full prompt tem-
plates are in Appendix B.2.
Citation Injector The Citation Injector oper-
ates as a post-processing step between the Solver
and the Veriﬁer, ﬁlling citation gaps in the rea-
soning trace by matching uncited claims against
the retrieved evidence via cross-encoder scoring.
Claims within the visual observations block are ex-
cluded from text-evidence matching and handled
separately via a pattern-based image citation strat-
egy. A citation is injected when the cross-encoder
score exceeds a tuned threshold τ= 0.4. Critically,
the injector replicates the exact evidence number-
ing used in the Solver’s prompt, guaranteeing that

any injected label refers to the same chunk the
Solver saw, preventing the citation misalignment
that would otherwise cause the Veriﬁer to ﬂag valid
claims as fabricated.
Veriﬁer The Veriﬁer receives the complete rea-
soning trace, all retrieved evidence, and the origi-
nal images, and performs claim-level fact-checking.
For each cited claim, it checks whether the refer-
enced evidence supports the stated fact, whether
any citation index is out of range, and whether
any image description contradicts the visual con-
tent. It produces a structured verdict with hallu-
cination severity (none, minor, major) and conﬁ-
dence (high, medium, low). Each detected hallu-
cination generates a feedback record injected into
the Extractor’s prompt on the next attempt (Ap-
pendix B.3).
3.2 Feedback Loop and Retry Logic
Major hallucinations always trigger a retry; mi-
nor hallucinations trigger a retry only when ver-
iﬁer conﬁdence is medium or low. A veriﬁed ver-
dict or any verdict after M=3attempts terminates
the pipeline. On retry, the Extractor receives the
feedback note, previously used queries, and full at-
tempt history, ensuring the evidence base changes
materially on each cycle.
4 Experimental Setup
Implementation. All conﬁgurations share iden-
tical hardware (3 ×NVIDIA A40 48 GB), tok-
enizers, and generation hyperparameters (Solver
T=0.3, Veriﬁer T=0.0, max 1024/2048 new to-
kens). Retrieval uses BM25 + FAISS dense re-
trieval with all-MiniLM-L6-v2 , cross-encoder re-
ranking, k=5chunks per sub-query, and up to 8
sub-queries per question, held ﬁxed across conﬁg-
urations.
Datasets. We evaluate on a stratiﬁed subsample
of ScienceQA ( Lu et al. ,2022) (n=1,000;52.3%
with images) spanning natural, social, and lan-
guage science, and on MMMU ( Yue et al. ,2024)
(n=500, 30 subjects across dev/validation ) for
generalization. The ScienceQA budget keeps the
95% Wilson CI for accuracy within ±3.1pp and
gives>80%power to detect a 6 pp gap between
conﬁgurations at α=0.05, matching our ablation
resolution. MMMU preserves ≥16items per sub-
ject. Details in Appendix E.
Metrics. No single existing metric jointly re-
wards correct, well-cited, attributable, and
evidence-grounded answers. CaVeScore combines
ﬁve complementary signals:
CaVeScore =w1·Acc+w2·CitePrec +w3·CiteRec
+w4·AIS+w5·Grounding (1)where Accis exact-match correctness; CitePrec
is mean NLI entailment between each cited chunk
and its claim; CiteRec is the fraction of factual
sentences carrying ≥1citation; AISis the frac-
tion of reasoning steps entailed by retrieved evi-
dence; and Grounding is the maximum entail-
ment between cited evidence and the ﬁnal an-
swer. We set (w1,...,w 5)=(0.4,0.2,0.2,0.1,0.1),
prioritising correctness while requiring claims be
both supported and cited. Weight sensitivity is in
§C.4.1; formal deﬁnitions of all 23 metrics are in
Appendix C.
Models. The pipeline uses ﬁve models. The
Extractor (Qwen2.5-7B-Instruct ( Team,2024),
4-bit NF4 quantised) generates retrieval sub-
queries. The Retriever combines dense em-
beddings from all-MiniLM-L6-v2 (Reimers and
Gurevych ,2019) with cross-encoder re-ranking
viams-marco-MiniLM-L6-v2 . The Solver , the
VLM that produces the chain-of-thought answer
over images and evidence, is LLaVA-CoT ( Xu
et al. ,2025). The Citation Injector reuses
thems-marco reranker to attach citations to
uncited claims. The Veriﬁer , the largest model
in the stack, is Qwen2.5-VL-32B-Instruct ( Bai
et al.,2025); its scale is deliberate, since § 5.4
shows the feedback loop’s value is bottlenecked
by veriﬁer capacity. At evaluation time only,
cross-encoder/nli-deberta-v3-base (He et al. ,
2023) scores entailment for CitePrec, AIS, and
Grounding. Full hyperparameters in Appendix E.
Ablation Conﬁgurations. We test whether
each pipeline stage contributes measurably to
CaVeScore with three conﬁgurations that progres-
sively add components (Table 1).
Conﬁg. Ext. Ret. Sol. Cit.I. Ver.
Solver-Only ∅ ∅ /check∅ ∅
Ret.-Solver /check /check /check ∅ ∅
CaVe-VLM-CoT /check/check/check/check/check
Table 1: Ablation conﬁgurations. /check= active;
∅= removed.
Solver-Only receives the question, choices, and
images but no retrieved evidence, generating CoT
from parametric knowledge alone. Retriever-
Solver adds hybrid retrieval and web search but
omits the Citation Injector, Veriﬁer, and feedback
loop, isolating the contribution of the veriﬁcation
machinery. CaVe-VLM-CoT is the full pipeline.
5 Results
The results are discussed below:
5.1 Overall Results
Table 2reports all metrics on both datasets. The
central diagnostic is the gap between accuracy

Metric ScienceQA MMMU ∆ Metric ScienceQA MMMU ∆ExtractorCoverage 0.682±0.201 0.501±0.203−0.181
SolverAccuracy 0.871±0.335 0.552±0.498−0.319
Speciﬁcity 0.999±0.006 0.987±0.111−0.012 Text Cite Prec. 0.322±0.375 0.120±0.241−0.202
Hit Rate 0.304±0.460 0.082±0.256−0.222 QI Cite Prec. 0.238±0.252 0.280±0.371 +0 .042
Subquery Count 7.491±0.715 7.632±1.169 +0.141 QI Cite Coverage 0.713±0.499 0.282±0.372−0.431RetrieverRecall@2 0.518±0.277 0.085±0.176−0.433 Citation Precision 0.543±0.399 0.325±0.358−0.218
Precision@2 0.150±0.273 0.050±0.159−0.100 Citation Recall 0.241±0.235 0.197±0.320−0.044
MRR 0.126±0.230 0.027±0.103−0.099 AIS 0.408±0.304 0.153±0.201−0.255
NDCG@2 0.123±0.228 0.027±0.103−0.096 Hallucination Rate 0.592±0.304 0.835±0.224 +0 .243VeriﬁerDecision Correct 0.525±0.500 0.670±0.471 +0.145 Grounding Score 0.197±0.343 0.163±0.296−0.034
Halluc. Det. Correct 0.309±0.462 0.234±0.424−0.075 Is Grounded 0.220±0.413 0.180±0.381−0.040
Conﬁdence Approp. 0.494±0.500 0.460±0.499−0.034 CaVeScore 0.566±0.2210.357±0.256−0.209
Feedback Quality 0.274±0.446 0.329±0.470 +0.055
Table 2: Full metric summary on ScienceQA ( n=1,000) and MMMU ( n=500, 30 subjects), reported
as mean ±std. For binary indicators (e.g. Accuracy, Hit Rate, Is Grou nded), std is the Bernoulli
standard deviation/radicalbig
p(1−p)across the sample; continuous metrics (CitePrec, AIS, CaVe Score) report
the empirical standard deviation. ∆= MMMU −ScienceQA.
Conﬁguration Acc. AIS CitePrec CaVe ∆CaVe
Solver-Only 0.730 0.125 0.000 0.314 —
Retriever-Solver 0.704 0.318 0.249 0.404 ↑0.090
CaVe-VLM-CoT 0.871 0.408 0.543 0.566↑0.252
Table 3: Ablation results on ScienceQA ( n=1,000).Bold = best per column. CaVeScore is the primary
ranking criterion. ∆CaVe is the gain over Solver-Only .
and grounding: on ScienceQA, accuracy ( 87.1%)
far exceeds AIS ( 40.8%), meaning a substantial
share of correct answers rely on parametric knowl-
edge rather than retrieved evidence. On MMMU,
this decoupling intensiﬁes; retrieval and ground-
ing metrics (Recall@2, AIS, CitePrec) collapse far
more steeply than accuracy ( ∆column), the sig-
nature of a corpus-coverage failure rather than a
reasoning failure. The VLM generalises across do-
mains; the retrieval corpus does not.
5.2 Cross-Dataset Generalisation
The pipeline was applied to MMMU without re-
training, prompt changes, or architectural modiﬁ-
cations. Table 2decomposes the cross-dataset gap
by stage. Retrieval collapses (Hit Rate −22.2pp,
Recall@2 −43.3pp), reﬂecting that the ScienceQA
lecture corpus contains little content relevant to
MMMU’s 30 university-level disciplines (e.g., clini-
cal medicine, mechanical engineering, art history).
With retrieval eﬀectively absent, the Solver falls
back on parametric knowledge, consistent with the
elevated hallucination rate ( +24.3pp). Yet accu-
racy degrades only −31.9pp and stays well above
the random baseline,showing the underlying VLM
still contributes useful parametric knowledge whenretrieval fails.
The pattern across metrics is itself diagnostic. The
retrieval and grounding metrics (Recall@2, AIS,
CitePrec) collapse much further than accuracy, the
signature of a corpus-coverage failure rather than
a reasoning failure. A response-level metric would
surface a single “MMMU accuracy drop” number;
the stage-level decomposition identiﬁes knowledge-
base coverage as the bottleneck, not query formula-
tion or reasoning architecture, and points directly
to corpus expansion as the actionable ﬁx.
5.3 Ablations
Table 3reports the four headline metrics across the
three conﬁgurations on the ScienceQA sample.
Moving from Solver-Only toRetriever-
Solver reduces accuracy by 2.6points (the re-
triever occasionally surfaces distracting evidence)
but adds +19.3points of AIS and +24.9of cita-
tion precision: retrieval converts parametric claims
into attributable, evidence-backed ones, driving
CaVeScore up by +9.0points. Moving from
Retriever-Solver to the full CaVe-VLM-CoT
adds another +16.7points of accuracy, +9.0of
AIS, and +29.4of citation precision, for a fur-

ther+16.2CaVeScore gain. The veriﬁcation loop
not only improves grounding (as expected) but
also substantially lifts accuracy, because the cal-
ibrated 32B Veriﬁer issues more reliable rejection
signals that drive targeted re-retrieval and surface
evidence missed on the ﬁrst pass.
5.4 Discussion
The single most consequential design choice is the
scale of the Veriﬁer (Table 4). Under the 8B model
(Qwen2.5-VL-8B), Decision Correctness was 9.9%,
well below the 50%chance rate for a binary ver-
dict, and Feedback Quality was 0%on every evalu-
ated sample: the feedback loop was architecturally
sound but empirically inert. At 32B, Decision Cor-
rectness rises to 52.5%and Feedback Quality to
27.4%, and the accuracy gain from adding the Ver-
iﬁer over Retriever-Solver jumps from +4.3to
+16.7points. This conﬁrms that the loop’s value
was previously bottlenecked by the quality-control
capacity of the smaller model, not by the loop’s de-
sign: a veriﬁer that cannot reliably tell grounded
from ungrounded reasoning produces feedback that
cannot steer re-retrieval.
Metric 8B 32B
Decision Correctness 9.9% 52 .5%
Feedback Quality 0.0% 27 .4%
∆Acc vs. Ret.-Solver +4.3pp+16.7pp
Table 4: Impact of Veriﬁer scale on veriﬁcation
quality and upstream accuracy. ∆Acc is the accu-
racy gain over Retriever-Solver (70.4%).
Residual failure modes. Two bottlenecks re-
main well characterised. First, veriﬁer calibration
is improved but not complete: Decision Correct-
ness is52.5%(just above chance). The dominant
residual failure mode is INCONCLUSIVE rejec-
tion handling, where the Veriﬁer ﬂags a problem
but cannot name the correct answer, overriding
a Solver that was in fact right. Second, retrieval
coverage on out-of-domain data is thin: MMMU’s
8.2%Hit Rate is the mechanism through which the
cross-dataset accuracy gap appears. Both are con-
crete, stage-identiﬁed failure modes that response-
level accuracy alone cannot expose, and both point
to actionable ﬁxes (veriﬁer scale or ﬁne-tuning, and
corpus expansion, respectively).
Ablation ordering is preserved under all six alter-
native CaVeScore weightings ( ∆range+0.20to
+0.30; full table in Appendix C.4.1).
6 Conclusion
We introduced CaVe-VLM-CoT, a ﬁve-stage
reﬂection-based agentic-RAG framework built onthe premise that hallucination is a grounding
failure, not a generation defect, and must be
caught structurally rather than mitigated through
prompting alone. By tying every reasoning step
to a veriﬁable citation and routing structured fail-
ure signals back to the Extractor when that link
breaks, it converts veriﬁcation from a passive post-
hoc ﬁlter into an active corrective mechanism.
The results both validate this commitment and
sharpen its limits. The pipeline transfers to
MMMU’s college-level reasoning across 30 disci-
plines with no architectural or prompt changes,
yet stage-level metrics expose a sharp retrieval-
coverage drop (Hit Rate 30.4%→8.2%) that
response-level accuracy would conceal, and abla-
tions conﬁrm each stage contributes measurably
to CaVeScore under all six alternative weight-
ings. The hallucination rate ( 59.2%on ScienceQA,
83.5%on MMMU) and residual veriﬁer miscalibra-
tion reveal that correctness and grounding remain
partially decoupled: the system reaches right an-
swers for partially wrong reasons, a failure mode
aggregate accuracy cannot expose. To our knowl-
edge, CaVe-VLM-CoT is the ﬁrst framework to
make veriﬁer calibration measurable at this gran-
ularity and expose it as a primary bottleneck in
citation-grounded multimodal reasoning.
7 Future Work
The evaluation points to three directions. First,
Veriﬁer rejection precision : at 32B, Decision
Correctness reaches 52.5%(from9.9%), but a RE-
JECTED verdict with Verified Answer: IN-
CONCLUSIVE still cannot reliably override a
correct Solver; recovering the Solver’s answer in
these cases, rather than defaulting to rejection,
would remove most residual false rejections with-
out retraining. Whether the remaining gap is a
matter of scale or of how veriﬁcation is supervised
is open. Second, the visual retrieval gap : image
evidence does not yet carry the citation-grounding
guarantees of retrieved text. Region-proposal
models for citation-relevant crops and cross-modal
rerankers that score image-claim pairs as reliably
as text could close it. Third, domain cover-
age, the dominant bottleneck for cross-domain
transfer: extending beyond multiple-choice QA to
open-ended generation, longer chains, and domain-
speciﬁc corpora (e.g., medical imaging, legal anal-
ysis) would test whether the architectural commit-
ments hold where failure modes and evidence types
diﬀer.

8 Limitations
Several limitations should be acknowledged. First,
the Veriﬁer remains the binding constraint on the
feedback loop. Upgrading to Qwen2.5-VL-32B
substantially improves calibration improving De-
cision Correctness from 9.9% to 52.5% and Feed-
back Quality from 0% to 27.4% but both remain
below ideal, and Hallucination Detection Correct-
ness is only 30.9%. The dominant failure mode
persists: the Veriﬁer ﬂags a problem but cannot
name the correct answer, occasionally overriding a
Solver that was in fact correct. Further scale or
task-speciﬁc ﬁne-tuning would likely help.
Second, image-based retrieval remains un-
solved. Both sliding-window patch retrieval
and grounding-DINO region proposals degraded
performance relative to text-only retrieval, forc-
ing us to pass question images directly to the
Solver. Visual evidence therefore cannot carry the
same citation-grounding guarantees as retrieved
text, leaving a modality gap in the attribution
framework.
Third, both evaluation datasets used in our ex-
periments (ScienceQA and MMMU) are multiple-
choice. Whether these results generalise to open-
ended generation, domain-speciﬁc corpora with
proprietary knowledge bases (e.g., clinical or legal
reasoning), or substantially longer reasoning chains
remains open. Relatedly, the CaVeScore weights
encode a speciﬁc design priority; our sensitivity
analysis (Section C.4.1) conﬁrms ablation rankings
are stable across seven conﬁgurations, though op-
timal weighting may vary by domain.
9 Broader Impact Statement
CaVe-VLM-CoT aims to make vision-language rea-
soning more transparent and veriﬁable by enforc-
ing citation grounding at every reasoning step. In
knowledge-intensive domains such as scientiﬁc edu-
cation, medical imaging, and legal analysis, tracing
each claim to a speciﬁc piece of evidence is a pre-
requisite for responsible deployment. By surfacing
hallucination at the stage level rather than hiding
it behind aggregate accuracy, the framework gives
practitioners a way to locate and address failure
modes before they reach end users. Some risks
remain. A well-cited answer is not necessarily a
correct one, and citation markers may lead users
to over-trust outputs without checking the cited
sources themselves. The evaluation suite, though
more informative than accuracy alone, inherits the
limitations of the component models it relies on,
so its scores should be read as diagnostic signals
rather than guarantees. The system also depends
on web search for evidence, which exposes it to
misinformation or bias in external sources. We re-
lease the evaluation framework and metric deﬁni-tions so the community can apply stage-level di-
agnostic evaluation to other multimodal reasoning
systems, and we encourage future work to stress-
test CaVeScore across diverse domains and failure
modes.
Use of Large Language Models
In accordance with transparency requirements, we
disclose the use of large language models during
the preparation of this manuscript. Claude (An-
thropic) was used as a writing aid for drafting,
paraphrasing, and editing prose in the manuscript
text. All technical content, experimental design,
system architecture, evaluation framework, metric
deﬁnitions, and reported results are the original
intellectual contribution of the authors. The au-
thors reviewed and edited all LLM-assisted text
and take full responsibility for the content of this
publication.
References
Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam
Altman, Shyamal Anadkat, and 1 others. 2023.
Gpt-4 technical report. Technical report.
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu,
Margaret Mitchell, Dhruv Batra, C Lawrence Zit-
nick, and Devi Parikh. 2015. Vqa: Visual question
answering. In Proceedings of the IEEE interna-
tional conference on computer vision , pages 2425–
2433.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil,
and Hannaneh Hajishirzi. 2024. Self-RAG: Learn-
ing to retrieve, generate, and critique through self-
reﬂection. In Proceedings of the 12th International
Conference on Learning Representations (ICLR) .
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng
Wang, Shijie Wang, Jun Tang, and 1 others.
2025. Qwen2.5-vl technical report. arXiv preprint
arXiv:2502.13923 .
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong
He, Zongbo Han, Zheng Zhang, and Mike Zheng
Shou. 2024. Hallucination of multimodal large
language models: A survey. arXiv preprint
arXiv:2404.18930 .
Changin Choi, Wonseok Lee, Jungmin Ko, and
Wonjong Rhee. 2025. Multimodal iterative RAG
for knowledge-intensive visual question answering.
arXiv preprint arXiv:2509.00798 .
Charles Corbiere, Simon Roburin, Syrielle Montar-
iol, Antoine Bosselut, and Alexandre Alahi. 2025.
Retrieval-based interleaved visual chain-of-thought
in real-world driving scenarios. arXiv preprint
arXiv:2501.04671 .

Gordon V. Cormack, Charles L. A. Clarke, and
Stefan Buettcher. 2009. Reciprocal rank fusion
outperforms condorcet and individual rank learn-
ing methods . InProceedings of the 32nd Interna-
tional ACM SIGIR Conference on Research and
Development in Information Retrieval , SIGIR ’09,
pages 758–759. ACM.
Wenliang Dai, Junnan Li, Dongxu Li, Anthony
Tiong, Junqi Zhao, Weisheng Wang, Boyang Li,
Pascale N Fung, and Steven Hoi. 2023. Instruct-
blip: Towards general-purpose vision-language
models with instruction tuning. Advances in
neural information processing systems , 36:49250–
49267.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeﬀ Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2025. The faiss library. IEEE Transactions
on Big Data .
Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi
Chen. 2023. Enabling large language models to
generate text with citations. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 6465–6488.
Pengcheng He, Jianfeng Gao, and Weizhu Chen.
2023. DeBERTav3: Improving deBERTa using
ELECTRA-style pre-training with gradient-disen-
tangled embedding sharing . InThe Eleventh Inter-
national Conference on Learning Representations .
Bin Ji, Huijun Liu, Mingzhe Du, and See-Kiong
Ng. 2024. Chain-of-thought improves text gener-
ation with citations in large language models. In
Proceedings of the AAAI Conference on Artiﬁcial
Intelligence , volume 38, pages 18345–18353.
Jeﬀ Johnson, Matthijs Douze, and Hervé Jégou.
2019. Billion-scale similarity search with gpus.
IEEE transactions on big data , 7(3):535–547.
Seongyun Lee, Seungone Kim, Sue Park, Gee-
wook Kim, and Minjoon Seo. 2024a. Prometheus-
vision: Vision-language model as a judge for ﬁne-
grained evaluation. In Findings of the Association
for Computational Linguistics: ACL 2024 , pages
11286–11315.
Tony Lee, Haoqin Tu, Chi Heem Wong, Wen-
hao Zheng, Yiyang Zhou, Yifan Mai, Jos-
selin Somerville Roberts, Michihiro Yasunaga,
Huaxiu Yao, Cihang Xie, and Percy Liang. 2024b.
VHELM: A holistic evaluation of vision language
models. Advances in Neural Information Process-
ing Systems , 37:140632–140666.
Patrick Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Na-
man Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. 2020. Retrieval-augmented genera-
tion for knowledge-intensive NLP tasks. In Ad-
vances in Neural Information Processing Systems
(NeurIPS) , volume 33, pages 9459–9474.Junnan Li, Dongxu Li, Caiming Xiong, and
Steven Hoi. 2022. BLIP: Bootstrapping language-
image pre-training for uniﬁed vision-language un-
derstanding and generation. In Proceedings of the
39th International Conference on Machine Learn-
ing (ICML) , pages 12888–12900.
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang,
Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evalu-
ating object hallucination in large vision-language
models. In Proceedings of the 2023 conference on
empirical methods in natural language processing ,
pages 292–305.
Haotian Liu, Chunyuan Li, Qingyang Wu, and
Yong Jae Lee. 2023. Visual instruction tuning. Ad-
vances in neural information processing systems ,
36:34892–34916.
Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng
Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei
Yang, Hang Su, Jun Zhu, and Lei Zhang. 2024.
Grounding DINO: Marrying DINO with grounded
pre-training for open-set object detection. In Pro-
ceedings of the European Conference on Computer
Vision (ECCV) . Springer.
Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu,
Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord,
Peter Clark, and Ashwin Kalyan. 2022. Learn
to explain: Multimodal reasoning via thought
chains for science question answering. In Ad-
vances in Neural Information Processing Systems
(NeurIPS) , volume 35, pages 2507–2521.
Kenneth Marino, Mohammad Rastegari, Ali
Farhadi, and Roozbeh Mottaghi. 2019. OK-VQA:
A visual question answering benchmark requir-
ing external knowledge. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , pages 3195–3204.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike
Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer,
Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023.
Factscore: Fine-grained atomic evaluation of fac-
tual precision in long form text generation. In
Proceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing , pages
12076–12100.
Nils Reimers and Iryna Gurevych. 2019. Sen-
tence-BERT: Sentence embeddings using Siamese
BERT-networks . InProceedings of the 2019 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP) , pages 3982–3992. Associa-
tion for Computational Linguistics.
Stephen E Robertson and Steve Walker. 1994.
Some simple eﬀective approximations to the 2-
poisson model for probabilistic weighted retrieval.
InSIGIR’94: Proceedings of the Seventeenth An-
nual International ACM-SIGIR Conference on Re-
search and Development in Information Retrieval,
organised by Dublin City University , pages 232–
241. Springer.
Anna Rohrbach, Lisa Anne Hendricks, Kaylee

Burns, Trevor Darrell, and Kate Saenko. 2018. Ob-
ject hallucination in image captioning. In Proceed-
ings of the 2018 Conference on Empirical Methods
in Natural Language Processing (EMNLP) , pages
4035–4045.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
ﬂexion: Language agents with verbal reinforce-
ment learning. In Advances in Neural Information
Processing Systems (NeurIPS) .
Qwen Team. 2024. Qwen2.5 technical report.
arXiv preprint arXiv:2412.15115 .
Junyang Wang, Yuhang Wang, Guohai Xu, Jing
Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang
Xu, Ming Yan, Ji Zhang, and 1 others. 2023.
AMBER: An LLM-free multi-dimensional bench-
mark for MLLM hallucination evaluation. arXiv
preprint arXiv:2311.07397 .
Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi,
Quoc V Le, and Denny Zhou. 2022. Chain-of-
thought prompting elicits reasoning in large lan-
guage models. In Advances in Neural Information
Processing Systems (NeurIPS) , volume 35, pages
24824–24837.
Tsung-Han Patrick Wu, Heekyung Lee, Jiaxin Ge,
Joseph Gonzalez, Trevor Darrell, and David Chan.
2026. Generate, but verify: Reducing hallucina-
tion in vision-language models with retrospective
resampling. Advances in Neural Information Pro-
cessing Systems , 38:65749–65777.
Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao
Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and
Chunyuan Li. 2025. LLaVA-Critic: Learning to
evaluate multimodal models. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , pages 13618–13628.
Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing
Song, Lichao Sun, and Li Yuan. 2025. Llava-COT:
Let vision language models reason step-by-step. In
Proceedings of the IEEE/CVF International Con-
ference on Computer Vision , pages 2087–2098.
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu
Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens,
Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong
Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming
Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu,
Wenhao Huang, and 3 others. 2024. MMMU:
A massive multi-discipline multimodal under-
standing and reasoning benchmark for expert
AGI. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition
(CVPR) , pages 9556–9567.
Di Zhang, Jingdi Lei, Junxian Li, Xunzhi Wang,
Yujie Liu, Zonglin Yang, Jiatong Li, Weida Wang,
Suorong Yang, Jianbo Wu, and 1 others. 2025a.
Critic-V: VLM critics help catch vlm errors in
multimodal reasoning. In Proceedings of theIEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 9050–9061.
Ruohong Zhang, Bowen Zhang, Yanghao Li, Hao-
tian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang,
Ruoming Pang, and Yiming Yang. 2025b. Im-
prove vision language model chain-of-thought rea-
soning. In Proceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 1631–1662.

Appendix
A Methodology Details
A.1 Corpus Construction
Our primary evaluation uses ScienceQA ( Lu et al. ,
2022), a large-scale multimodal benchmark span-
ning natural science, social science, and language
science questions, each paired with multiple-choice
answer options, optional context images, lecture
text, and annotated solutions. As part of prepro-
cessing, images are normalised to a uniform spa-
tial representation of 224x224; a pre-trained cap-
tioning model ( Li et al. ,2022) and optical char-
acter recognition (OCR) are applied to produce
textualised descriptions of each image’s content;
and subject, topic, and skill metadata are stored
alongside each example. A critical design choice
is that lecture text, hints, and subject metadata
arenotinjected into the Extractor prompt; they
reside in the knowledge base and must be surfaced
by well-formed retrieval queries. If this metadata
were provided directly to the Extractor, the system
could trivially match against it without demon-
strating genuine retrieval capability, artiﬁcially in-
ﬂating Hit Rate and Recall metrics. By withhold-
ing it, we ensure the pipeline is evaluated under
realistic inference-time conditions where such an-
notations are unavailable. Full preprocessing de-
tails are in Appendix E.
A.2 Knowledge Base Construction
The local knowledge base is built from the Sci-
enceQA lecture corpus and indexed using two com-
plementary indices maintained in parallel:
• Adense vector index (FAISS ( Douze et al. ,
2025)) that encodes text chunks into ﬁxed-
length embeddings for semantic similarity
search, capturing meaning-level matches even
when surface words diﬀer.
• Asparse term-frequency index (BM25
(Robertson and Walker ,1994)) for keyword
matching, retrieving chunks that share exact
terms with the query but may lie far apart in
embedding space.
This dual-index design captures evidence that ei-
ther approach would miss in isolation. The two
indices are fused at retrieval time via the Recip-
rocal Rank Fusion ( Cormack et al. ,2009), so the
fusion strategy can be tuned independently of in-
dex construction.
B Prompt Engineering
B.1 Extractor Prompt
The Extractor receives only the question, answer
choices, and image-derived context (BLIP captions
and OCR text). No lecture text, hints, or subjectlabels are provided. Its task is to output a struc-
tured list of 3-12 word retrieval queries covering
three complementary families, (i)deﬁnitional and
conceptual queries about key terms in the question;
(ii)choice-discriminating queries that help surface
evidence distinguishing among the answer options;
and(iii)comparative queries that relate multiple
choices against each other.
The prompt terminates with an open bracket to
prime structured list generation. Four in-context
examples cover distinct reasoning types: geo-
graphic comparison, grammar classiﬁcation, phys-
ical science, and title capitalisation, each demon-
strating a query set that spans at least one def-
initional query and one query per answer choice.
Output parsing operates in three fallback tiers,
full structured list parsing (primary path), par-
tial recovery from truncated output (secondary),
and line-by-line extraction from plain numbered-
list output (last resort). Parsed queries are vali-
dated for length (2-12 words), content-bearing to-
kens, and non-repetition of the question stem. A
deterministic choice-discriminating augmentation
step then appends one query per answer choice,
guaranteeing retrieval coverage for every candidate
regardless of model output. On retry iterations,
structured Veriﬁer feedback is spliced between the
question block and the opening bracket, with pre-
viously used queries listed explicitly to discourage
repetition.
Listing 1: Extractor prompt skeleton (abbrevi-
ated).
Generate search queries to retrieve the
evidence needed to answer a
multiple-choice question. Your queries
will be run against a knowledge
base AND a web search engine.
OUTPUT FORMAT: a structured list of query
strings, nothing else.
Each query: 3-12 words, plain keywords or
short phrases, no question marks.
[Four few-shot examples omitted for
brevity]
Question: {question}
Choices: {choices}
{image_context}[
B.2 Solver Prompts
The Solver maintains two prompt templates se-
lected at runtime based on whether the question
has associated images. Both enforce a rigid four-
block output format.
Text-only questions. The model structures its
response as:
1.<SUMMARY> states the core problem without

factual claims;
2.<REASONING> each step must follow “According
to [Text Evidence N], ...” or“From domain
knowledge, ...” ;
3.<CONCLUSION> restates the answer without in-
troducing new facts.
Vision questions. An<OBSERVATIONS> section
is prepended, requiring the model to describe each
image using mandatory [Question Image N] ci-
tations before reasoning begins. A cross-encoder
consistency check runs post-parse: for each answer
choice the cross-encoder scores the pair (reason-
ing text, candidate answer), and if the highest-
scoring choice diﬀers from the stated answer by
more than 0.5 score units, the answer is silently
corrected. This guards against a failure mode com-
mon in smaller VLMs where the model reasons to
the correct conclusion but writes the wrong letter.
Citation rules. The model is instructed explic-
itly and negatively: it is strictly better to have
zero citations than to cite evidence that does not
support a claim. Domain knowledge claims must
be preﬁxed with “From domain knowledge, ...”
only after consulting all retrieved evidence. A low-
temperature retry ( T= 0.1) ﬁres if the initial gen-
eration lacks text citations despite substantive re-
trieved evidence, or lacks visual observations de-
spite images being present.
B.3 Veriﬁer Prompt
The Veriﬁer is designed as a strict fact-checker op-
erating over the complete reasoning trace, the full
retrieved evidence set (formatted with the same
citation labels as the Solver’s prompt), and the
original question images. Its task is claim-level,
not response-level. For each cited claim it checks
whether the referenced evidence chunk actually
supports the stated fact, whether any citation in-
dex is out of range, and whether any image descrip-
tion contradicts what is visually present. Common
knowledge inferences, paraphrases of the question,
and partial evidence support are explicitly listed as
non-hallucinations to prevent over-ﬂagging of legit-
imate reasoning.
The output format is strictly structured and parsed
via regex:
Listing 2: Veriﬁer structured output format.
Hallucination Check: [NONE DETECTED |
MINOR HALLUCINATIONS | MAJOR
HALLUCINATIONS]
[If hallucinations detected:]
Claim: "<exact quote from solver>"
Issue: fake citation | not in evidence
| misrepresented | fabricated
Evidence: <what the evidence actually
says, or "doesn’t exist">Final Verdict: [VERIFIED | REJECTED]
Confidence: [HIGH | MEDIUM | LOW]
Verified Answer: [A/B/C/D/E |
INCONCLUSIVE]
A critical alignment requirement: the Veriﬁer re-
constructs the evidence index using identical ﬁl-
tering and capping logic to the Solver, capping at
ﬁve chunks per sub-query and ten total. If these
rules diﬀer by even one entry, a citation valid in
the Solver’s prompt resolves to a diﬀerent chunk
in the Veriﬁer’s context, causing correct citations
to be ﬂagged as fabricated and triggering spurious
retry cycles.
C Evaluation Framework
Metrics. For the primary results we foreground
four headline metrics:
•Accuracy : exact match of the predicted
answer letter ˆyto the gold index y∗, av-
eraged over Nexamples: Accuracy =
1
N/summationtextN
i=1/notforces{ˆyi=y∗
i}.
•AIS: fraction of factual reasoning sentences
s∈Sfor which at least one retrieved evidence
chunke∈ Eentailssunder an NLI model:
AIS=1
|S|/summationtext
s∈S/notforces{maxe∈EPNLI(entail|
e,s)> τAIS}, withτAIS= 0.35.
•Citation Precision : mean NLI entail-
ment score of each (evidence, claim) citation
pair in the Solver’s response: CitePrec =
1
|C|/summationtext
(e,c)∈Cscore NLI(e,c), where score NLI∈
{0,0.5,1}for contradiction/neutral/entail-
ment.
•CaVeScore : a weighted composite of Accu-
racy, Citation Precision, Citation Recall, AIS,
and Grounding.
Accuracy alone combines correct answers achieved
through hallucinated reasoning with those that are
genuinely grounded. CaVeScore is therefore our
primary ranking criterion, as it jointly rewards cor-
rectness andtransparent attribution.
CaVeScore diﬀers from prior evaluation approaches
in that it jointly measures answer correctness, ci-
tation precision and recall, step-level attribution
(AIS), and evidence grounding within a single com-
posite. Accuracy and BLEU/ROUGE capture only
the ﬁrst dimension; ALCE and AIS target cita-
tion and attribution respectively but do not span
both; and holistic VLM judges such as LLaVA-
Critic ( Xiong et al. ,2025) and VHELM ( Lee et al. ,
2024b ) score responses as a whole rather than
decomposing quality across these axes. A full
dimension-by-dimension comparison is provided in
Appendix F. We evaluate CaVe-VLM-CoT along
four axes, Extractor quality, Retriever quality,

Stage Model Size Role
Extractor Qwen2.5-7B-Instruct ( Team,2024) 7B (4-bit) Sub-query generation
Retriever all-MiniLM-L6-v2 ( Reimers and Gurevych ,2019) 22M Dense embedding
Re-ranker ms-marco-MiniLM-L-6-v2 22M Cross-encoder rera nk
Solver LLaVA-CoT ( Xu et al. ,2025) 11B VLM reasoning + CoT
Citation Injector ms-marco-MiniLM-L-6-v2 22M Citation at tachment
Veriﬁer Qwen2.5-VL-32B-Instruct ( Bai et al. ,2025) 32B Verdict + feedback
Evaluation only (not in pipeline):
NLI scorer nli-deberta-v3-base ( He et al. ,2023) 184M CitePrec, AIS, Grounding
Table B1: Models used in each pipeline stage. Sizes approxim ate; all hyperparameters in Appendix E.
Solver quality, and Veriﬁer quality. Throughout
this section, let Qdenote the set of sub-queries gen-
erated by the Extractor, φ(·)an embedding func-
tion (all-MiniLM-L6-v2 ), topK(q)theKhighest-
ranked chunks returned for sub-query q,y∗the gold
answer index, and ˆythe predicted answer letter.
PNLI(entailment |p,h)is the entailment probabil-
ity from a natural language inference model with
premise pand hypothesis h.Edenotes the full
set of retrieved evidence chunks, and Ecited⊆ E
the subset explicitly cited in the Solver’s response.
/notforces{·}is the indicator function.
C.1 Extractor Metrics
Coverage Score. Measures how thoroughly the
generated sub-queries cover the key terms in the
input. Let Tbe the set of non-stopword alphabetic
tokens (length ≥3) extracted from the question
and answer choices, and Sthe concatenated text
of all sub-queries:
Coverage =|{t∈ T:t∈S}|
|T |(2)
A score of 1.0 indicates every key term appears in
at least one sub-query.
Speciﬁcity Score. Quantiﬁes whether individ-
ual sub-queries are precise enough to retrieve tar-
geted evidence. For each sub-query qwith word
countw, a length score ℓ(q)∈ {0.3,0.6,1.0}is
assigned based on thresholds ( ℓ= 0.3ifw <3;
ℓ= 1.0ifw >15;ℓ= 0.6otherwise), with a
ﬁxed penalty of 0.2 subtracted when qbegins with
a generic preamble (e.g., “What is”):
Speciﬁcity =1
|Q|/summationdisplay
q∈Qmax/parenleftbig
0, ℓ(q)−penalty(q)/parenrightbig
(3)
Hit Rate. A binary indicator that ﬁres when
any retrieved chunk, across all sub-queries, is se-
mantically close to the gold answer. Speciﬁcally,
it requires at least one top- Kchunkcfor some
sub-query qto exceed a cosine similarity threshold
θ= 0.5with the embedding of the concatenated
question and gold answer:Hit=/logicalordisplay
q∈Q/logicalordisplay
c∈topK(q)/notforces/braceleftbig
cos/parenleftbig
φ(q+gold), φ(c)/parenrightbig
> θ/bracerightbig
(4)
C.2 Retriever Metrics
Recall@ K.Fraction of sub-queries for which at
least one top- Kchunk is relevant, where relevance
is deﬁned as either exceeding a cosine similarity
threshold of 0.35 with the gold-answer embedding
or containing the gold answer as a substring:
Recall@ K=1
|Q|/summationdisplay
q∈Q/notforces/braceleftbigg
max
c∈topK(q)cos/parenleftbigφ(q+gold),
φ(c)/parenrightbig>0.35∨gold⊆c/bracerightbigg
(5)
Precision@ K.Fraction of all retrieved chunks
(across all sub-queries) that are relevant to the gold
answer, using a stricter cosine threshold of 0.5:
Precision@ K=|{(q,c) :c∈topK(q),cos(φ(gold),φ(c))>0.5}|
|Q| ×K
(6)
MRR. Mean Reciprocal Rank over sub-queries.
Letr(q)be the rank of the ﬁrst chunk whose cosine
similarity with the gold-answer embedding exceeds
0.5 (r(q) =∞if no chunk qualiﬁes):
MRR=1
|Q|/summationdisplay
q∈Q1
r(q)(7)
NDCG@ K.Normalized Discounted Cumula-
tive Gain over the top- Kchunks. A chunk at rank
iis relevant (rel i= 1) if its cosine similarity to the
gold-answer embedding exceeds 0.5, and Rdenotes
the total number of relevant chunks:
DCG@K=K/summationdisplay
i=1reli
log2(i+1),
IDCG@K=R/summationdisplay
i=11
log2(i+1),
NDCG@ K=DCG@K
IDCG@K(8)

C.3 Solver Metrics
Accuracy. Binary correctness of the predicted
answer. The Solver outputs a letter ˆy∈
{A,B,C,...}, which is compared against the gold
indexy∗(zero-indexed):
Accuracy =/notforces{ord(ˆy)−ord(A) =y∗}(9)
Text Citation Precision (NLI-based). Mea-
sures how well each text-based citation is sup-
ported by its linked evidence. Let Ctextbe the set of
(evidence chunk, cited claim) pairs extracted from
the Solver’s response. Each pair receives a score of
1.0 (entailment), 0.5 (neutral), or 0.0 (contradic-
tion) from the NLI model:
TextCitePrec =1
|Ctext|/summationdisplay
(e, c)∈ CtextscoreNLI(e,c) (10)
Question Image Citation Precision. Frac-
tion of image citations in the Solver’s response that
reference a valid image index. Let CQIbe the set
of image citations and |imgs|the number of images
associated with the question:
QICitePrec =|{c∈ CQI: 1≤c.id≤ |imgs|}|
|CQI|(11)
Citation Recall. Measures whether factual sen-
tences in the Solver’s chain-of-thought are backed
by at least one citation. Let Calldenote all citations
and|factual_sentences |the number of factual sen-
tences identiﬁed in the response:
CiteRecall = min/parenleftbigg
1.0,|Call|
max(1,|factual_sentences |)/parenrightbigg
(12)
AIS (Attributable to Identiﬁed Sources).
Fraction of factual sentences s∈Sin the Solver’s
response for which at least one evidence chunk in
Eprovides entailment support above a threshold
of 0.35:
AIS=1
|S|/summationdisplay
s∈S/notforces/braceleftbigg
max
e∈EPNLI(entailment |e, s)>0.35/bracerightbigg
(13)
Evidence Grounding Score. Measures how
strongly the ﬁnal answer ais entailed by the cited
evidence. It takes the maximum entailment prob-
ability over all cited chunks:
Grounding = max
e∈EcitedPNLI(entailment |e, a)(14)
CaVeScore (Composite). A weighted compos-
ite metric that balances answer correctness with
citation and grounding quality:
CaVeScore =w1·Accuracy +w2·CitePrecision
+w3·CiteRecall +w4·AIS
+w5·Grounding (15)The weights w1,...,w 5satisfywi≥0,/summationtext
iwi= 1,
and are set to (0.4,0.2,0.2,0.1,0.1)in all experi-
ments; see Section C.4.1 for a sensitivity analysis.
The weights reﬂect a design priority. Correctness
is necessary but insuﬃcient; a response must also
be veriﬁably grounded.
C.4 Veriﬁer Quality Metrics
Ground Truth Construction. To evaluate the
Veriﬁer, we construct a pseudo ground-truth accep-
tance label from upstream metrics. A response is
deemed acceptable if it is correct, well-cited, and
attributable:
gt_accept = (Accuracy = 1.0)∧(CitePrecision ≥0.7)
∧(AIS≥0.5)(16)
Decision Correctness. Binary agreement be-
tween the Veriﬁer’s Verified /Rejected verdict
and the pseudo ground-truth label gt_accept.
Hallucination Detection Correctness. Bi-
nary agreement between the Veriﬁer’s hallucina-
tion ﬂag and the NLI-derived hallucination signal,
which ﬁres when the hallucination rate exceeds 0.3.
Conﬁdence Appropriateness. Evaluates
whether the Veriﬁer’s conﬁdence level ( High,
Medium ,Low) is consistent with the evidence
quality. High is appropriate only when citations
are valid andreasoning is grounded; Low is
appropriate only when either condition fails;
Medium is always considered appropriate.
Feedback Quality. Measures whether the Ver-
iﬁer’s structured feedback correctly identiﬁes real
issues and provides actionable guidance. For each
of three issue dimensions d(accuracy, citation qual-
ity, grounding), let δd∈ {−1,0,1}score true pos-
itives (+1), true negatives ( 0), and false posi-
tives/negatives ( −1):
IssueAcc =1
3/summationtext
dδd+ 1
2∈[0,1],
FeedbackScore = 0.7·IssueAcc + 0.3·/notforces{actionable }(17)
Feedback quality is capped at 0.5 when no real
issues exist, penalising spurious rejections.
C.4.1 CaVeScore Weight Sensitivity
The CaVeScore weights in Equation 1encode a
speciﬁc design priority. To verify that our ablation
rankings are not artifacts of this particular weight-
ing, we re-score all per-sample metric components
under seven alternative weight conﬁgurations:
the published default (0.4/0.2/0.2/0.1/0.1),
accuracy-heavy (0.6/0.1/0.1/0.1/0.1),
citation-heavy (0.2/0.3/0.3/0.1/0.1), uni-
form (0.2/0.2/0.2/0.2/0.2), AIS-heavy

(0.3/0.15/0.15/0.25/0.15), grounding-heavy
(0.3/0.15/0.15/0.15/0.25), and recall-skewed
(0.35/0.1/0.35/0.1/0.1). No re-inference is re-
quired; each conﬁguration re-weights the ﬁve
pre-computed component scores and produces a
new composite CaVeScore per sample.
Two observations are worth highlighting. First,
the absolute CaVeScore shifts as expected when
weights change: conﬁgurations that up-weight ac-
curacy (where our system is strongest, 87.1%) score
higher in the aggregate, while those that up-weight
the grounding-related metrics score lower. This
movement is not a weakness of the metric, it is
the diagnostic signal we designed CaVeScore to
produce. Second, and more importantly, the rel-
ative ordering of the three conﬁgurations is pre-
served under every weighting, and the gap between
the full pipeline and the Solver-Only baseline
ranges from +0.201(accuracy-heavy) to +0.302
(citation-heavy). The gap widens precisely under
the weightings that stress grounding quality, which
is the regime where the full pipeline’s retrieval and
veriﬁcation machinery matters most. The ablation
conclusions therefore do not depend on the speciﬁc
choice of weights in Equation 1.
D Experimental Results
Table 2reports all 23 metrics as mean ±standard
deviation over 1,000ScienceQA examples.
E Implementation Details
Dataset preprocessing. All ScienceQA
images are resized to 224×224 pix-
els and converted to RGB. Captions are
generated using BLIP ( Li et al. ,2022)
(Salesforce/blip-image-captioning-base )
in batches of eight. OCR is performed using
Tesseract. The processed dataset is serialised as
both CSV and JSON, with image paths updated
to resized copies, so every pipeline stage reads
from the same augmented representation.
MMMU dataset details. MMMU ( Yue et al. ,
2024) covers 30 subjects (e.g., medicine, engi-
neering, art history, economics) with 500ques-
tions across devandvalidation splits. It dif-
fers from ScienceQA in three respects: (i) every
question includes at least one ﬁgure (chart, di-
agram, microscopy image, or blueprint), making
visual reasoning mandatory; (ii) questions require
university-level domain expertise, testing whether
the retrieval pipeline can surface specialised evi-
dence; and (iii) no built-in lecture corpus exists,
so the Retriever relies more heavily on web search.
Images are downloaded from HuggingFace Hub via
thedatasets library and resized to 224×224
pixels. BLIP captions are generated identically
to ScienceQA; OCR is omitted as MMMU im-ages are predominantly diagrams and charts. The
explanation ﬁeld (present only in the devsplit) is
mapped to the lecture column for knowledge base
indexing; validation split rows receive an empty
lecture ﬁeld. MMMU’s FAISS index is stored in
a separate subdirectory ( indexes/mmmu/ ) to avoid
overwriting the ScienceQA index. Answer letters
(A-E) are converted to zero-indexed integers. No
pipeline code changes were required.
Knowledge base indexing. Dense embeddings
useall-MiniLM-L6-v2 (Reimers and Gurevych ,
2019) (384 dimensions), normalised to unit length
and stored in a FAISS ﬂat index ( Johnson et al. ,
2019) for exact inner-product search. The sparse
index uses BM25Okapi with whitespace tokenisa-
tion. Hybrid fusion uses RRF with k= 60.
Retrieval hyperparameters. Dense
and sparse retrieval each return top-
3candidates per sub-query before fu-
sion. The cross-encoder reranker
(cross-encoder/ms-marco-MiniLM-L-6-v2 )
operates over a top-50 candidate pool; the top two
local corpus chunks are reserved unconditionally
and up to k=5total chunks are passed per sub-
query (10 maximum across all sub-queries). Web
search returns up to two snippets per sub-query
plus one additional snippet per choice-augmented
variant.
Model conﬁguration. The Extrac-
tor (Qwen2.5-7B-Instruct) is loaded in 4-
bit NF4 quantisation via bitsandbytes
(unsloth/Qwen2.5-7B-Instruct-bnb-4bit ),
with greedy decoding ( T=0.0) and a max-
imum of 512 new tokens. The Solver
(LLaVA-CoT ( Xu et al. ,2025), an 11B-
parameter Llama-3.2-Vision model ﬁne-tuned
for chain-of-thought reasoning; checkpoint
zhangsongbo365/Llama-3.2V-11B-cot-nf4 ) uses
greedy decoding ( T=0.0) and a maximum of 1024
new tokens; citation-retry calls use T=0.1. The
Veriﬁer (Qwen2.5-VL-32B-Instruct) uses greedy
decoding ( T=0.0) and a maximum of 2048 new
tokens, ensuring deterministic verdicts. All three
models run on dedicated NVIDIA A40 48 GB
GPUs (one model per GPU) with PyTorch’s math
SDPA attention backend.
NLI evaluation. All NLI-based metrics use
cross-encoder/nli-deberta-v3-base with
apply_softmax=True . Entailment threshold for
AIS is 0.35; and precision metrics, 0.5.
Observability. Every pipeline stage is instru-
mented via OpenInference-compliant spans captur-
ing model inputs, outputs, token counts, latencies,
and per-stage evaluation metrics, enabling post-
hoc per-example trace analysis.

Conﬁg waccwcprec wcrec waiswgnd Solver-Only Retriever-Solver CaVe-VLM-CoT Gap ↑
default 0.40 0.20 0.20 0.10 0.10 0.314 0.404 0.566 +0.251
accuracy-heavy 0.60 0.10 0.10 0.10 0.10 0.460 0.508 0.661 +0.201
AIS-heavy 0.30 0.15 0.15 0.25 0.15 0.265 0.371 0.510 +0.245
recall-skewed 0.35 0.10 0.35 0.10 0.10 0.278 0.362 0.504 +0.226
grounding-heavy 0.30 0.15 0.15 0.15 0.25 0.263 0.356 0.489 +0.227
uniform 0.20 0.20 0.20 0.20 0.20 0.191 0.311 0.452 +0.261
citation-heavy 0.20 0.30 0.30 0.10 0.10 0.168 0.300 0.470 +0.302
Gap↑=CaVe-VLM-CoT −Solver-Only .
Table C2: CaVeScore weight sensitivity analysis across the three ablation conﬁgurations on ScienceQA
(n=1,000). Each row re-scores every per-sample component under a diﬀ erent weight vector. The absolute
values shift, but the ordering Solver-Only <Retriever-Solver <CaVe-VLM-CoT is preserved
under every weighting, conﬁrming that the ablation ranking s are robust to reasonable weight perturbation.
F Comparison with Existing
Evaluation Frameworks
Table F3contrasts the quality dimensions captured
by CaVeScore against those of existing evaluation
approaches. We mark a dimension as /checkwhen it is
directly measured by the framework, ∼when par-
tially captured, and ∅when not measured. Judge-
ments are made on each framework’s primary or
headline metric; frameworks often report auxiliary
numbers that touch adjacent dimensions, but these
are not the axes on which they are designed to rank
systems.

Metric / FrameworkAnswer
Correct.Citation
PrecisionCitation
RecallStep-level
AttributionEvidence
Grounding
Accuracy / Exact Match /check ∅ ∅ ∅ ∅
BLEU / ROUGE ∼ ∅ ∅ ∅ ∅
ALCE ( Gao et al. ,2023) ∅ /check /check ∅ ∅
AIS (Ji et al. ,2024) ∅ ∅ ∅ /check ∅
LLaVA-Critic ( Xiong et al. ,2025)∼ ∅ ∅ ∅ ∼
VHELM ( Lee et al. ,2024b ) /check ∅ ∅ ∅ ∅
CaVeScore (ours) /check /check /check /check /check
Table F3: Comparison of evaluation metrics across quality d imensions. /check= directly measured; ∼=
partially captured; ∅= not measured.