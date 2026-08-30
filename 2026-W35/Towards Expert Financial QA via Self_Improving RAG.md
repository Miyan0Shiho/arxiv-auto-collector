# Towards Expert Financial QA via Self-Improving RAG

**Authors**: Junjie Xiong, Shawheen Ghezavat, Aum Hirpara

**Published**: 2026-08-27 07:01:41

**PDF URL**: [https://arxiv.org/pdf/2608.26706v1](https://arxiv.org/pdf/2608.26706v1)

## Abstract
Expert-level financial question answering requires both grounded verification to catch numeric hallucinations and audit trails for regulatory compliance, attributes that standard single-pass RAG systems lack. We take a step toward this goal with Self-Improving RAG, a framework that decomposes document QA into three specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator with feedback-driven self-correction. When the Judge Agent scores an answer below a dynamic threshold, the system triggers retry with escalated strategies: broader retrieval, more careful prompting, and relaxed acceptance criteria. We evaluate on FinanceBench (SEC filing QA), where Self-Improving RAG achieves 86% oracle-guided accuracy (measuring agreement with gold answers) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers through targeted retry. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results without dynamic routing, providing full interpretability. Every decision is logged with confidence scores, enabling the audit trails required for regulated financial applications.

## Full Text


<!-- PDF content starts -->

Published as a conference paper at ICLR 2026
TOWARDSEXPERTFINANCIALQAVIASELF-
IMPROVINGRAG
Junjie Xiong
University of California, BerkeleyShawheen Ghezavat
California Polytechnic State University
Aum Hirpara
Hofstra University
ABSTRACT
Expert-level financial question answering requires bothgrounded verificationto
catch numeric hallucinations andaudit trailsfor regulatory compliance, attributes
that standard single-pass RAG systems lack. We take a step toward this goal
with Self-Improving RAG, a framework that decomposes document QA into three
specialized agents (Retrieval, Reasoning, and Judge) coordinated by an orchestrator
with feedback-driven self-correction. When the Judge Agent scores an answer
below a dynamic threshold, the system triggers retry with escalated strategies:
broader retrieval, more careful prompting, and relaxed acceptance criteria. We
evaluate on FinanceBench (SEC filing QA), where Self-Improving RAG achieves
86% oracle-guided accuracy (measuring agreement with gold answers) with a
36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect answers through
targeted retry. A key finding is that a fixed retrieval pipeline with judge-driven retry
achieves strong results without dynamic routing, providing full interpretability.
Every decision is logged with confidence scores, enabling the audit trails required
for regulated financial applications.
1 INTRODUCTION
Retrieval-augmented generation (RAG) has become the standard paradigm for knowledge-intensive
question answering (Lewis et al., 2021). However, as RAG systems are deployed in high-stakes
domains such as finance (Islam et al., 2023; Tai et al., 2025), a critical limitation emerges: conventional
single-pass pipelines lack the ability to recognize and correct their own failures (Asai et al., 2023;
Yan et al., 2024).
Consider a financial analyst querying SEC filings to extract quarterly revenue figures. A single-pass
RAG system retrieves documents, generates an answer, and returns with no mechanism to verify
correctness. For financial professionals, an incorrect answer is worse than no answer, and a system
without anaudit trailis indistinguishable from a guess.
The Walled Garden Constraint.Financial applications impose a crucial constraint that distin-
guishes them from general-domain QA: retrieval must remain within authorized document corpora.
Unlike systems that can fall back to web search when initial retrieval fails (Yan et al., 2024), financial
QA systems operate in a “walled garden” where data governance policies prohibit external informa-
tion sources. This constraint eliminates a common recovery mechanism and demands alternative
approaches to self-correction. Importantly, this closed-domain constraint also serves as anagent
governance mechanism: by limiting retrieval scope, we ensure every answer traces to authorized,
auditable sources, a key requirement for responsible deployment of agentic systems in regulated
industries.
In finance specifically, three additional challenges compound: (1)numeric reasoningover tables
and financial statements, (2)temporal filteringrequiring understanding of fiscal years and reporting
periods, and (3)entity disambiguationamong ticker symbols, subsidiaries, and corporate name
changes.
1
arXiv:2608.26706v1  [cs.CL]  27 Aug 2026

Published as a conference paper at ICLR 2026
Figure 1: Three-agent architecture: Retrieval, Reasoning, and Judge agents coordinated by an
Orchestrator. When score s1< τ, our system retries with escalated retrieval ( k:10→20→30 ) and
adapted prompting until accepted or budget exhausted.
Our Approach.We proposeSelf-Improving RAG, a framework that decomposes retrieval-
augmented generation into three specialized agents coordinated by an orchestrator with feedback-
driven self-correction. TheRetrieval Agentanalyzes queries, extracts entities, and selects from
available retrieval pipelines with escalation based on attempt number. TheReasoning Agentgener-
ates answers with explicit citations and adapts its prompting strategy based on prior attempt outcomes.
TheJudge Agentevaluates answer quality against dynamic thresholds and decides whether to accept
or request retry.
The key insight is that when initial retrieval or generation fails, the system canescalaterather
than return low-quality answers: retrieve more documents, prompt more carefully, and accept
marginally lower confidence when effort has been expended. This contrasts with prior adaptive
retrieval approaches that focus on initial routing without recovery mechanisms (Jeong et al., 2024;
Asai et al., 2023).
Our contributions toward expert-level financial QA are:
1.Structured Self-Correction for Document QA: Unlike prior single-model self-
reflection (Madaan et al., 2023), we introducetargeted escalationacross retrieval, gen-
eration, and evaluation stages, with specialized agents diagnosing whether failure originated
in retrieval or reasoning
2.Grounded Verification: A Judge that performs entailment checking against retrieved
evidence, with programmatic numeric verification for financial figures
3.Audit-First Design: Every agent decision logged with provenance, confidence scores, and
reasoning traces, enabling compliance review for regulated industries
On FinanceBench, Self-Improving RAG achieves 86% LLM Judge accuracy with a 36.4% Lazarus
Rate, recovering nearly 4 in 10 initially incorrect answers. A key finding is that a fixed retrieval
pipeline with judge-driven retry achieves strong results without dynamic routing, providing full
interpretability.
2 RELATEDWORK
Self-Correction in Language Models.The emerging paradigm of Agentic RAG (Singh et al.,
2025) embeds autonomous agents into retrieval pipelines to enable self-correction and adaptive
2

Published as a conference paper at ICLR 2026
reasoning beyond single-pass systems. Reflexion (Shinn et al., 2023) uses verbal reinforcement
learning across episodes; Self-Refine (Madaan et al., 2023) achieves 15–20% gains via single-model
critique-refine loops. Self-RAG (Asai et al., 2023) trains models to emit reflection tokens, while
CRAG (Yan et al., 2024) triggers corrective retrieval on failure. Unlike Reflexion, we correctwithin
a single session; unlike Self-Refine, we usespecialized agentswith domain-specific verification;
unlike Self-RAG, we require no fine-tuning. CRAG’s web search fallback violates financial “walled
garden” constraints. Self-correction can also be learned via RL (Kumar et al., 2024), which defines a
correction ratemetric conceptually similar to our Lazarus Rate; our approach achieves self-correction
without RL training, using heuristic escalation and specialized judging. Crucially, our Judge Agent
diagnoses whether failure originated in retrievalorreasoning and triggers targeted retry, unifying
both correction modalities.
Adaptive Retrieval and Document QA.Adaptive-RAG (Jeong et al., 2024) routes queries to
different pipelines based on complexity. Unlike approaches focusing oninitialselection, we combine
routing withjudge-driven retry: if the first attempt fails, the system escalates rather than returning
low-quality answers. A fixed pipeline with judge-driven retry provides a simpler alternative to learned
routing, critical for explainability in regulated domains. LLM Judge (Zheng et al., 2023) provides
scalable quality assessment; recent surveys (Li et al., 2024) highlight judge biases that motivate our
separate Judge Agent design. Agent-as-a-Judge (Zhuge et al., 2024) achieves 90% human agreement
via multi-turn evaluation, supporting our iterative feedback approach.
Multi-Agent Coordination.AutoGen (Wu et al., 2023) and MetaGPT (Hong et al., 2024) pioneered
agentic frameworks for multi-turn LLM coordination. Unlike these general-purpose systems, we tailor
agent roles specifically to document QA with finance-domain constraints and audit requirements.
Positioning: The Walled Garden Constraint.A key distinction of our work is theclosed-domain
constraint: retrieval must remain within authorized documents, precluding web search fallbacks that
violate financial data governance policies. Table 1 summarizes how Self-Improving RAG relates to
prior methods across four dimensions critical for financial applications.
Method Self-Correct No Fine-tune Closed-Domain Audit Trail
Self-RAG (Asai et al., 2023)✓×✓×
CRAG (Yan et al., 2024)✓ ✓× ×
Adaptive-RAG (Jeong et al., 2024) ×✓ ✓×
Reflexion (Shinn et al., 2023)✓ ✓× ×
Self-Improving RAG (Ours)✓ ✓ ✓ ✓
Table 1: Comparison with related self-correction and adaptive retrieval methods. Self-Improving
RAG is the only approach that combines within-session self-correction, requires no task-specific
fine-tuning, operates strictly within authorized document corpora (“walled garden”), and provides
full audit trails for regulatory compliance.
Self-RAGtrains models to emit special reflection tokens that trigger self-correction, but requires
fine-tuning on curated (input, output, reflection) triples, which limits domain adaptation.CRAGuses
web search as a fallback when initial retrieval fails, which violates data governance requirements in
regulated industries.Adaptive-RAGlearns to route queries to different pipelines but lacks a retry
mechanism for recovery. Our approach combines the benefits of self-correction (like Self-RAG and
CRAG) with the training-free deployment (like CRAG and Adaptive-RAG) while maintaining strict
closed-domain operation and audit trails.
3 METHOD: SELF-IMPROVINGRAG
Problem Setting.Given a natural language query q∈ Q and an authorized corpus D=
{d1, . . . , d n}of financial documents, produce an answer asupported by evidence E⊆ D . We
assume aclosed-domainsetting where retrieval must remain within authorized documents, a regula-
tory constraint precluding web search fallbacks. The system may attempt up to B+ 1 total attempts
3

Published as a conference paper at ICLR 2026
(where Bis the retry budget), accepting when utility exceeds threshold or returning the best answer
if budget exhausts.
3.1 PRELIMINARIES ANDNOTATION
The system state at attempttis:
St=⟨E t, at,st, τt⟩(1)
where Et⊆ D is retrieved evidence, atis the candidate answer, st∈[0,1]3is the quality vector, and
τtis the acceptance threshold. Three specialized agents, Retrieval ( R), Reasoning ( G), and Judge
(J), are coordinated by an Orchestrator (Figure 1).
Retrieval Agent.The retrieval operator maps query and corpus to evidence:
Et=R(q,D;ϕt)(2)
where ϕt={k t, σt}parameterizes top- kretrieval and RSE expansion. We use a hybrid pipeline
(dense + BM25 + reranking). On retry, we retrieve 10 additional documents per attempt, up to a
maximum of k=30 . Relevant Segment Extraction (RSE) activates only on the final attempt. Table 2
shows the escalation configuration.
Attempt top k initial k RSE
1 (Standard) 103×Off
2 (Escalated) 204×Off
3 (Maximum) 306×On
Table 2: Retrieval Agent escalation strategies. On retry, the agent retrieves more documents and
progressively enables Relevant Segment Extraction (RSE).
Routing Heuristics.The rule-based router selects retrieval pipelines based on query characteristics:
•If the question contains a recognized ticker symbol or company name →hybrid retrieval
with metadata filtering
•If the question requests numerical comparison or computation →hybrid retrieval with
metadata filtering and reranking (precision-focused)
• If the question is open-ended or exploratory→hybrid retrieval with broad recall
• Default fallback→semantic-only retrieval
Entity recognition uses a simple gazetteer of S&P 500 tickers plus regex patterns for fiscal year
mentions. This lightweight approach adds negligible latency ( <10ms) while achieving routing
decisions that empirically match LLM-based classifiers.
Finance Lexicon & Normalization.Financial queries frequently use shorthand and domain
jargon (e.g., “top line,” “YoY ,” “capex,” “diluted EPS”) that do not lexically match SEC filing
terminology. We therefore maintain a lightweight finance lexicon consisting of (i)entity aliases
and identifiers(tickers, company names, subsidiaries), (ii)metric canonicalizations(synonym sets
mapping “top line” →revenue, “SG&A” →operating expenses), and (iii)unit/period normalizers
(thousands/millions/billions; fiscal-year and quarter expressions). The Retrieval Agent uses the
lexicon for query normalization and synonym expansion prior to hybrid retrieval, while the Judge
performs strict numeric comparison (see Section 4 for discussion of unit normalization opportunities).
See Appendix A.3 for sample entries.
Reasoning Agent.Given query and evidence, the generator samples:
at∼Pθ(a|q, E t, πt)(3)
where θdenotes LLM parameters and πtranges over prompting regimes (STANDARD →CONSER-
VATIVE →DETAILED) that escalate with t. Retrieved documents are formatted with explicit source
boundaries to enable citation extraction.
4

Published as a conference paper at ICLR 2026
Judge Agent.The judge maps(q, a t, Et)to a quality vector:
st=J(q, a t, Et) = [µ g, µc, µn]⊤∈[0,1]3(4)
separatinggrounding µg(evidence entailment),completeness µc(query coverage), andnumeric
faithfulnessµ n. We aggregate into scalar utility:
Ut=w⊤st=wgµg+wcµc+wnµn,w∈R3
≥0,1⊤w= 1(5)
whereµ g≡J ground, µc≡J complete , µn≡J numeric (formalized in Appendix B).
Finance constraint.We weight numeric faithfulness heavily ( wn= 0.5 ,wg= 0.3 ,wc= 0.2 ),
penalizing numeric errors even when answers appear fluent. Numeric faithfulness uses strict set
coverage:
µn(at, Et) =1 
|N(a t)\N(E t)|= 0
(6)
where N(·) extracts normalized numeric quantities. This catchesnear-misshallucinations by requir-
ing every number ina tbe explicitly supported byE t.
Orchestrator.The orchestrator accepts or retries based on:
τt= max(τ 0−λ(t−1), τ min),D t=1(U t≥τt)(7)
withτ0=0.5 ,λ=0.1 ,τmin=0.3 . This decay reflects a precision-coverage tradeoff: early attempts
demand high confidence, while later attempts accept marginal answers. If no attempt is accepted,
return the best:
a⋆=aˆt,ˆt= arg max
t∈{1,...,T}Ut.(8)
All decisions are logged with timestamps and reasoning traces for audit compliance.
Proposition 1(Convergence Rate).The probability of system failure after Tattempts decays multi-
plicatively, bounded by the conditional failure probability of each stage:
Pfail=TY
t=1P(D t= 0| S t−1).(9)
See Appendix B for the formal proof.
3.2 ORCHESTRATORALGORITHM
Algorithm 1 formalizes the orchestration loop. The key insight is that the system maintains thebest
answer seen so far, ensuring that retry never degrades output quality. When the Judge scores an
attempt below the dynamic threshold, the orchestrator triggers escalation: broader retrieval, more
careful prompting, and relaxed acceptance criteria.
The algorithm’s monotonic improvement guarantee is crucial for deployment: stakeholders can trust
that allowing more retries never produces worse answers, only potentially better ones with increased
latency. Table 7 in Appendix B summarizes notation.
4 EXPERIMENTS ANDRESULTS
4.1 EXPERIMENTALSETUP
We evaluate onFinanceBench(Islam et al., 2023), a benchmark of 150 SEC filing questions where
66% require numerical calculations. We set retry budget B=2 and initial threshold τ0=0.5 . Im-
plementation uses GPT-4o-mini for generation, BGE-large embeddings (Chen et al., 2025) with
ChromaDB, and cross-encoder reranking. The Judge combines LLM Judge evaluation with program-
matic numeric verification. Full details in Appendix E.
5

Published as a conference paper at ICLR 2026
Algorithm 1Self-Improving RAG Orchestrator
Require:Questionq, retry budgetB
1:best answer← ∅, best score←0
2:attempt←1
3:whileattempt≤B+ 1do
4:docs←RetrievalAgent.retrieve(q,attempt)
5:answer←ReasoningAgent.generate(q,docs,attempt)
6:score←JudgeAgent.evaluate(q,answer)
7:ifscore>best scorethen
8:best answer←answer
9:best score←score
10:end if
11:if¬JudgeAgent.should retry(score,attempt)then
12:break
13:end if
14:attempt←attempt+ 1
15:end while
16:returnbest answer, best score
Evaluation Protocol: Oracle-Guided vs. Deployment Modes. Critical limitation:We report
results under two evaluation protocols that reveal a significant gap. InDeployment mode(Table 5),
where the Judge operates blind without gold answers, we achieve only31%acceptance rate. This
reflects the fundamental challenge of quality estimation without ground truth. InOracle-Guided
mode(Table 3), where the Judge has gold-answer access for direct comparison with prior work,
we achieve86%accuracy, demonstrating the system’s potential when judge quality improves. The
36.4% Lazarus Rate is measured in deployment mode, showing the blind Judge still successfully
identifies and corrects a substantial fraction of failures, but stronger judge models (e.g., Claude Opus,
GPT-4o) could close this gap.
4.2 MAINRESULTS
Table 3 compares single-pass RAG against Self-Improving RAG on FinanceBench using oracle-
guided evaluation.
Configuration Correctness∆
Single-pass RAG 53%[45, 61]–
Self-Improving RAG86%[80, 91]+62.3%
Table 3: FinanceBench correctness with 95% bootstrap confidence intervals (oracle-guided evalua-
tion). Self-correction improves accuracy by detecting incomplete answers and triggering retry.
Performance by Question Type.Table 4 breaks down results by FinanceBench question category,
revealing that self-correction provides the largest gains on domain-relevant questions (+81.1%),
which often require synthesizing information across multiple document sections.
Question Type Single-Pass Self-Corr.∆
Metrics-generated (66%) 53% 82% +54.7%
Domain-relevant (22%) 53% 96%+81.1%
Novel-generated (12%) 53% 80% +50.9%
Overall 53% 86% +62.3%
Table 4: Correctness by question type. Domain-relevant questions benefit most from self-correction
(+81.1%), as these often have incomplete first-pass answers.
6

Published as a conference paper at ICLR 2026
Lazarus Rate: Measuring Resilience.We measure theLazarus Rateto quantify self-correction
effectiveness: the percentage of initially incorrect answers successfully corrected through retry. Let
W1={q:J 1(q)< τ correct}denote initially incorrect answers. The Lazarus Rate measures recovery:
LazarusRate=|{q∈ W 1:J2(q)≥τ correct}|
|W1|=P(correct 2|wrong1).(10)
On FinanceBench, the Lazarus Rate is36.4%: of 33 questions where the blind Judge triggered
retry (22% of total), 12 were successfully corrected. Appendix C provides detailed correction flow
analysis; Appendix D presents case studies and error taxonomy.
Correction Flow Visualization.Figure 2 visualizes the complete “life of a question” through our
self-correction pipeline, showing how 150 questions flow through the system.
Figure 2: Correction flow visualization. TheLazarus Raterepresents the proportion of initially
incorrect answers successfully corrected through retry.
Component Ablation.Table 5 reports ablations in deployment mode (blind judge, no gold an-
swers).
Configuration Blind Judge Acc.∆
Full System (B=2) 31%[24, 39]–
−Prompt Escalation 28%[21, 35]−10.6%
−Retrieval Escalation 30%[23, 37]−4.3%
−Deterministic Verify 32%[25, 39]+2.1%
Reduced Budget (B=1) 23%[17, 30]−25.5%
Table 5: Component ablation (deployment mode, no gold answers). Prompt escalation contributes
most; retry budget is crucial.
Removing prompt escalation causes the largest accuracy drop ( −10.6% ), indicating that enhanced
prompting on retry is the most valuable component. The retry budget comparison (B=1 vs B=2)
shows a substantial −25.5% drop, validating the importance of allowing multiple correction attempts.
The Deterministic Verification Anomaly.Interestingly, removing deterministic numeric verifica-
tionslightly improvesaccuracy ( +2.1% ), despite our design weighting numeric faithfulness heavily.
We investigated this counterintuitive result and identified the root cause: over-sensitive numeric
matching. The deterministic verifier flags answers as incorrect when numbers appear in slightly
different formats (e.g., “$394.3 billion” vs. “$394,300 million”) or when rounding differs by small
amounts. In these cases, a semantically correct answer triggers unnecessary retry, and the second
attempt may introduce new errors.
This finding highlights a gap between our design intent (prioritizing numeric accuracy) and implemen-
tation reality. Two directions for improvement emerge: (1) relaxing the numeric matching tolerance
7

Published as a conference paper at ICLR 2026
for format variations, and (2) implementing unit-aware normalization before comparison. We leave
these refinements to future work, noting that the difference ( +2.1% ) falls within our confidence
intervals and should be interpreted cautiously.
Judge Discrimination.The Judge’s retry decision is evaluated as a binary classifier:
TPR=P(RETRY|wrong),FPR=P(RETRY|correct).(11)
High TPR ensures failures trigger retry; low FPR avoids unnecessary overhead.
Note:The evaluation Judge (Table 3) has gold-answer access, while the production Judge performs
blind verification, avoiding circular self-evaluation.
4.3 DESIGNIMPLICATIONS
Our experiments use a fixed hybrid retrieval pipeline (dense + sparse with reranking), rather than
dynamic routing. This design choice reflects a key insight:the Judge Agent’s retry mechanism
provides a safety net that makes perfect initial retrieval unnecessary. When the first attempt fails,
escalation strategies (more documents, RSE segment merging) can recover.
This challenges the assumption that neural routing universally benefits RAG systems. For financial
QA where query characteristics are domain-specific (entity mentions, fiscal years), the self-correction
loop may be more valuable than perfect initial routing.
5 CONCLUSION
We presented Self-Improving RAG, a multi-agent framework that decomposes retrieval-augmented
generation into specialized agents with a self-correction feedback loop. The Judge Agent’s quality
assessment, combined with escalating retrieval and generation strategies, enables systematic recovery
from failures that single-pass systems cannot address.
Key Findings.On FinanceBench, Self-Improving RAG achieves 86% LLM Judge accuracy
(+62.3% over baselines) with a 36.4% Lazarus Rate, recovering nearly 4 in 10 initially incorrect
answers. A key finding is that a fixed retrieval pipeline with judge-driven retry achieves strong results
without dynamic routing, providing full interpretability. Among components, prompt escalation
contributes most, suggesting that encouraging careful reasoning is more valuable than retrieval
expansion alone.
Limitations.Several limitations warrant discussion.First, blind judge reliability is the main
bottleneck for deployment.Blind quality estimation without gold answers remains difficult; stronger
judges could narrow this gap. Second, the self-correction loop increases latency when retry is triggered
(15-25 seconds vs. 5-8 seconds for single-pass), making this system appropriate for analyst support
rather than real-time applications. Third, numeric exact-match accuracy does not improve with self-
correction, as our mechanism primarily recovers semantic incompleteness rather than extraction errors.
Fourth, our evaluation on n=150 questions limits statistical power for ablation comparisons; the
confidence intervals (11–15 points) mean differences like −10.6% and+2.1% should be interpreted
cautiously.
Future Work.Several directions merit exploration: (1) replacing heuristic threshold decay with
learned escalationvia reinforcement learning, (2) implementing unit-aware normalization to address
the deterministic verification anomaly, (3) extending the framework to other high-stakes domains
(legal, medical) requiring audit trails, and (4) leveraging explicit agent boundaries for human-in-the-
loop oversight.
Broader Impact.As LLMs are deployed in regulated industries, the demand for interpretable,
auditable AI systems will grow. Self-Improving RAG demonstrates that multi-agent architectures
can provide these properties while maintaining competitive performance. By logging every decision
with provenance and confidence scores, we enable the post-hoc analysis and regulatory compliance
that financial institutions require. We hope this work contributes to responsible AI deployment in
high-stakes domains.
8

Published as a conference paper at ICLR 2026
REFERENCES
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/
abs/2310.11511.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-embedding:
Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge
distillation, 2025. URLhttps://arxiv.org/abs/2402.03216.
D-Star AI. dsrag: High-quality retrieval through document-segment structuring. https://
github.com/D-Star-AI/dsRAG, 2024.
Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang,
Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin
Wu, and J ¨urgen Schmidhuber. Metagpt: Meta programming for a multi-agent collaborative
framework, 2024. URLhttps://arxiv.org/abs/2308.00352.
Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and Bertie Vidgen.
Financebench: A new benchmark for financial question answering, 2023. URL https://
arxiv.org/abs/2311.11944.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity, 2024.
URLhttps://arxiv.org/abs/2403.14403.
Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate
Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha
Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, and Aleksandra
Faust. Training language models to self-correct via reinforcement learning, 2024. URL https:
//arxiv.org/abs/2409.12917.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https:
//arxiv.org/abs/2005.11401.
Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun
Liu. Llms-as-judges: A comprehensive survey on llm-based evaluation methods, 2024. URL
https://arxiv.org/abs/2412.05579.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative
refinement with self-feedback, 2023. URLhttps://arxiv.org/abs/2303.17651.
Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and
Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL
https://arxiv.org/abs/2303.11366.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented
generation: A survey on agentic rag, 2025. URL https://arxiv.org/abs/2501.09136 .
Zhenghan Tai, Hanwei Wu, Qingchen Hu, Jijun Chi, Hailin He, Lei Ding, Tung Sum Thomas
Kwok, Bohuai Xiao, Yuchen Hua, Suyuchen Wang, Peng Lu, Muzhi Li, Yihong Wu, Liheng Ma,
Jerry Huang, Jiayi Zhang, Gonghao Zhang, Chaolong Jiang, Jingrui Tian, Sicheng Lyu, Zeyu
Li, Boyu Han, Fengran Mo, Xinyue Yu, Yufei Cui, Ling Zhou, and Xinyu Wang. Veritasfi: An
adaptable, multi-tiered rag framework for multi-modal financial question answering, 2025. URL
https://arxiv.org/abs/2510.10828.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and
Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation, 2023. URL
https://arxiv.org/abs/2308.08155.
9

Published as a conference paper at ICLR 2026
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation,
2024. URLhttps://arxiv.org/abs/2401.15884.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.
Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https://arxiv.org/
abs/2306.05685.
Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang
Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, Yangyang Shi,
Vikas Chandra, and J ¨urgen Schmidhuber. Agent-as-a-judge: Evaluate agents with agents, 2024.
URLhttps://arxiv.org/abs/2410.10934.
A EXTENDEDMETHODOLOGY
This appendix provides additional technical details on the Self-Improving RAG architecture.
A.1 RETRIEVALESCALATIONSTRATEGIES
Table 2 (in Section 3) shows the escalation configuration used on retry. These parameters reflect a
conservative-to-aggressive strategy: the initial attempt uses a focused context window ( k= 10 ) to
minimize noise, while subsequent attempts progressively expand recall.
A.2 ROUTINGHEURISTICS
The rule-based router operates as follows:
•If the question contains a recognized ticker symbol or company name →hybrid filter
(metadata filtering)
•If the question requests numerical comparison or computation →
hybrid filter rerank(precision-focused)
• If the question is open-ended or exploratory→hybrid(broad recall)
• Default fallback→semantic
Entity recognition uses a simple gazetteer of S&P 500 tickers plus regex patterns for fiscal year
mentions. This lightweight approach adds negligible latency ( <10ms) while achieving routing
decisions that empirically match LLM-based classifiers.
A.3 FINANCELEXICON
Table 6 shows sample entries from our finance lexicon used for query normalization and numeric
verification.
Canonical Synonyms / Aliases
Metrics
Revenue net sales, total revenues, top line, turnover
COGS cost of revenue, cost of sales
Operating income operating profit, income from operations
Free cash flow FCF, cash generated (CFO−capex)
EPS earnings per share, diluted EPS, basic EPS
Units & Periods
(in millions) in thousands, in billions
FY2023 fiscal year 2023, fiscal 2023
YoY year-over-year, y/y
QoQ quarter-over-quarter, sequential, q/q
Table 6: Sample entries from the finance lexicon for query expansion and unit normalization.
10

Published as a conference paper at ICLR 2026
A.4 PROMPTSTRATEGIES
The Reasoning Agent maintains three prompting strategies that vary in instruction specificity:
•Standard: Concise instructions emphasizing accuracy and citation
•Conservative: Additional instructions to acknowledge uncertainty when evidence is weak
•Detailed: Expanded instructions requiring step-by-step reasoning and explicit source attri-
bution
On retry, the agent escalates from standard to conservative to detailed, progressively encouraging
more careful reasoning.
A.5 CONFIDENCEESTIMATIONSIGNALS
The Reasoning Agent estimates answer confidence using heuristic signals:
• Presence of hedging language (“may”, “possibly”, “uncertain”)
• Refusal phrases (“cannot determine”, “not enough information”)
• Presence of specific numerical values (increases confidence)
• Answer length (extremely short answers indicate low confidence)
A.6 EVALUATIONSIGNALSTRUCTURE
The Judge produces a structured assessment rather than a single scalar:
•Grounding score∈[0,1]: Proportion of answer claims with explicit textual support
•Completeness score∈[0,1]: Whether all question components are addressed
•Numeric verification: Binary flag from programmatic extraction and comparison
•Confidence signals: Presence of hedging language, refusal phrases
The final score aggregates these signals, with numeric verification given highest weight for financial
queries.
A.7 RELEVANTSEGMENTEXTRACTION(RSE)
RSE (D-Star AI, 2024) merges adjacent high-scoring chunks from the same document into coherent
segments. Unlike query expansion techniques that generate hypothetical content, RSE operates
purely on retrieved documents: it identifies chunk boundaries, detects adjacency (same page or
consecutive chunks), and greedily selects segments that maximize relevance while respecting context
length budgets. This ensures all context provided to the Reasoning Agent comes from actual source
documents, which is critical for financial applications requiring strict groundedness.
A.8 DESIGN FORTRUSTWORTHINESS
Our multi-agent architecture incorporates several features aligned with responsible AI principles:
Audit Trails.All agent decisions are logged with timestamps, confidence scores, and reason-
ing traces. This enables post-hoc analysis of failure modes and supports regulatory compliance
requirements.
Uncertainty Signals.The Judge Agent’s quality score serves as an uncertainty estimate: low
scores indicate the system recognizes potential errors, triggering self-correction rather than returning
unreliable answers.
Human Oversight Points.The explicit agent boundaries create natural intervention points. A
human reviewer can inspect retrieved documents before generation, or override the Judge’s retry
decision.
11

Published as a conference paper at ICLR 2026
B FORMALDEFINITIONS
This section provides rigorous mathematical definitions for the agent functions and convergence
properties referenced in the main paper.
B.1 AGENTFUNCTIONSIGNATURES
LetQdenote the space of questions,Athe space of answers, andDthe document corpus.
Retrieval Agent.R:Q ×N→ P(D)returns the top-kdocuments relevant to a query:
R(q, k) =top-k({d∈ D:ϕ(q, d)> θ})(12)
where ϕ(q, d) is a relevance scoring function (combining dense and sparse signals) and θis a minimum
relevance threshold.
Reasoning Agent.Let Σ ={σ std, σcons, σdetail}denote the set of prompt strategies (standard,
conservative, detailed):
G:Q × P(D)×Σ→ A(13)
The agent generates an answer conditioned on the question, retrieved documents, and the current
prompt strategy.
Judge Agent. J:Q×A×P(D)→[0,1] computes a quality score with the following components:
Jground(a, D) =1
mmX
i=11[∃d∈D:NLI(d, c i) =ENTAIL](14)
Jcomplete (q, a) =1
nnX
j=11[addresses(a, p j)](15)
Jnumeric (a, D) =1[∀v∈ N(a) :∃v′∈ N(D),match(v, v′)](16)
where {ci}m
i=1are claims extracted from answer a,{pj}n
j=1are sub-questions parsed from q,N(·)
extracts numeric values, and match(v, v′)verifies exact numeric equivalence.
B.2 CONVERGENCEANALYSIS
The self-correction loop terminates in at most Tmax=B+ 1 iterations (where Bis the retry budget):
T= min{t≥1 :J t≥τtort=T max}(17)
With retry probabilityp 1=P(J 1< τ1)≈0.22(observed on FinanceBench), the expected cost is:
E[C] =c·(1 +p 1+p1p2)≈1.3c(18)
where cis the cost of a single attempt and p2≈p 1assumes similar retry probability on second
attempts.
Best-Answer Selection.The Orchestrator maintains the best answer across all attempts, ensuring
monotonic improvement in final output quality:
a∗= arg max
t∈{1,...,T}J(q, a t, Dt)(19)
This selection criterion guarantees that retry never degrades the final answer, even if later attempts
produce lower scores.
12

Published as a conference paper at ICLR 2026
B.2.1 PROOF OFPROPOSITION1 (CONVERGENCE)
Statement:The self-correcting process reduces failure probability multiplicatively with respect to
the number of attemptsT.
Proof. LetFtdenote the event that the system fails to produce an acceptable answer (i.e., Ut< τt)
at attemptt. The system terminates successfully at steptifD t= 1(success) occurs.
The total system failure Pfailoccurs only if the system fails ateveryattempt t∈ {1, . . . , T} . Thus,
we compute the joint probability:
Pfail=P(F 1∩F2∩ ··· ∩F T)(20)
By the probability chain rule, this joint distribution factorizes as:
P(F 1∩ ··· ∩F T) =P(F 1)·P(F 2|F1)·P(F 3|F1, F2)···P(F T|F1, . . . , F T−1)(21)
In our Markovian formulation (Eq. (1)), the state St−1encapsulates all relevant history. Thus, the
probability of failing at steptgiven previous failures equals:
P(F t|F1, . . . , F t−1) =P(D t= 0| S t−1)(22)
Substituting yields the multiplicative decay:
Pfail=TY
t=1P(D t= 0| S t−1)(23)
Since the probability of failure at any single stage is less than 1 (assuming non-zero success probabil-
ity),P fail→0exponentially asT→ ∞.
B.3 NOTATIONSUMMARY
Symbol Description
Q,A,DQuestion, answer, document spaces
R(q, k)Retrieval function returning top-kdocuments
G(q, D, σ)Generation function with prompt strategyσ
J(q, a, D)Judge scoring function
Jground, Jcomplete , Jnumeric Judge component scores
τt Dynamic threshold at attemptt
BRetry budget (maximum number of retries)
TActual number of attempts (termination time)
wg, wc, wn Component weights in Judge aggregation
ϕ(q, d)Document relevance scoring function
ΣSet of prompt strategies
Table 7: Notation summary for the Self-Improving RAG framework.
C ADDITIONALRESULTS
C.1 ABLATIONSTUDY
Table 8 shows the contribution of each component on FinanceBench.
C.2 CORRECTIONFLOWANALYSIS
Table 9 presents the complete self-correction flow on FinanceBench.
13

Published as a conference paper at ICLR 2026
Configuration FinanceBench
Full Self-Improving RAG 0.86
−Judge Agent (no retry) 0.53
Single-pass baseline 0.53
Table 8: Ablation study on FinanceBench. Each row removes one component.
Metric Count Rate
Total Questions 150 –
Confident (no retry) 117 78.0%
Triggered Retry (Judge flagged) 33 22.0%
Corrected by Retry 1236.4%(Lazarus Rate)
Still Wrong after Retry 21 63.6%
Table 9: Self-correction flow analysis on FinanceBench. The “Lazarus Rate” measures what per-
centage of initially incorrect answers were successfully corrected through retry. Note: The Lazarus
Rate measures improvement inLLM Judge scores(semantic correctness), not numeric exact-match.
Self-correction recovers semantically incomplete answers but does not improve numeric precision
(Table 3).
C.3 DISPROPORTIONATENUMERICGAIN
C.4 RETRIEVALPIPELINE
Our experiments use a fixed hybrid filter rerank pipeline for all questions, combining dense
embeddings (BGE-large) with sparse retrieval (BM25) and cross-encoder reranking. This design
choice prioritizes simplicity and reproducibility over dynamic routing.
The self-correction mechanism compensates for suboptimal initial retrieval: when the Judge identifies
a low-quality answer, the Retrieval Agent escalates to more aggressive strategies (higher k, RSE
segment merging). This “safety net” approach may be more practical than attempting perfect initial
routing.
C.5 CORRECTIONFLOWVISUALIZATION
Figure 2 (in the main paper) visualizes the complete “life of a question” through our self-correction
pipeline, showing how 150 questions flow through the system with the Lazarus Rate (36.4%)
representing successful corrections.
D CASESTUDIES ANDERRORANALYSIS
D.1 UNIT/SCALECONFUSIONERROR
Example: Unit/Scale Confusion Error
Question:What was Company X’s total revenue for FY2023?
Retrieved Context:“...total revenues of $X,XXX for the fiscal year ended December 31, 2023 (in
millions)...”
Model Answer (Initial):$X,XXX
Ground Truth:$X.X billion
Analysis:The model correctly extracted the numeric value but failed to apply the “in millions” unit
qualifier, producing an answer off by a factor of 1,000. The Judge Agent detected this inconsistency and
triggered retry. On the second attempt, the model correctly interpreted the unit context.
14

Published as a conference paper at ICLR 2026
Metric Single-Pass Self-Improving∆
Semantic Similarity 0.50 0.50 +0.0%
LLM Judge Accuracy0.53 0.86+62.3%
Table 10: Self-correction improves LLM Judge accuracy substantially while maintaining semantic
similarity. The Judge Agent’s ability to detect and correct errors provides meaningful recovery.
D.2 FAILUREMODEANALYSIS
To understand system limitations, we analyze cases where Self-Improving RAG fails to improve over
single-pass baselines:
•Retrieval ceiling: When relevant information is absent from the document corpus, escalated
retrieval cannot recover
•Arithmetic errors: Multi-step calculations accumulate rounding errors or apply incorrect
formulas
•Unit/scale confusion: Misinterpreting units (thousands vs. millions) or mixing absolute
values with percentages
•Temporal misalignment: Extracting figures from incorrect fiscal periods or conflating FY
with calendar year dates
•Judge miscalibration: The Judge Agent scores an incorrect answer highly, preventing
beneficial retry
•Hallucinated figures: The model generates specific numerical values not present in retrieved
context
E IMPLEMENTATIONDETAILS
E.1 DATASETS
FinanceBench(Islam et al., 2023) contains questions about publicly traded companies requiring
extraction and reasoning over SEC 10-K and 10-Q filings. Notably, 66% of FinanceBench questions
require numerical calculations.
E.2 FINANCE-SPECIFICCHALLENGES
FinanceBench questions present three challenges:
•Numerical Precision: Metrics-generated questions require exact extraction and calculation
from financial tables
•Temporal Context: Questions often specify fiscal periods that must be resolved to specific
filing dates
•Multi-Document Reasoning: Novel-generated questions may require synthesizing infor-
mation across multiple filings
E.3 BASELINES
We compare Self-Improving RAG against single-pass baselines:
•Semantic: Dense retrieval with BGE-large embeddings
•Hybrid: Combined dense and BM25 sparse retrieval
•Hybrid + Filter: Hybrid retrieval with metadata filtering
•Hybrid + Filter + Rerank: Full pipeline with cross-encoder reranking
E.4 EVALUATIONMETRICS
Semantic Similarity.Cosine similarity between generated and gold answers using sentence-
transformers.
15

Published as a conference paper at ICLR 2026
Numeric Verification.For numerical answers, we extract and compare numeric values.
LLM Judge Score.GPT-4o-mini rates answer quality on a 0-1 scale (Zheng et al., 2023).
Lazarus Rate (Correction Rate).The percentage of initially incorrect answers successfully
corrected through retry:
Lazarus Rate=|{q:wrong1(q)∧correct 2(q)}|
|{q:wrong1(q)}|
E.5 MODELS ANDCONFIGURATION
Models.We use GPT-4o-mini as the primary generation model.
Retrieval.Documents are chunked with 512-token windows and 50-token overlap. We use BGE-
large-en-v1.5 embeddings (Chen et al., 2025) stored in ChromaDB. The reranker is BGE-reranker-
large. Default retrieval returnsk= 10documents.
Agent Configuration.The Retrieval Agent escalates from standard ( k= 10 ) to escalated ( k= 20 ),
and finally to maximum recall ( k= 30 with RSE). The Judge Agent uses threshold τ= 0.5 for
Attempt 1, decreasing toτ= 0.4for subsequent attempts.
Retry Budget.Maximum retries set to 2 (up to 3 total attempts).
E.6 EFFICIENCY
Self-Improving RAG introduces overhead from multiple agent calls and potential retries:
•Single-pass latency:∼5–8 seconds per question
•Multi-Agent (no retry):∼8–12 seconds per question (+50% overhead)
•Multi-Agent (with retry):∼15–25 seconds per question (when retry triggered)
This system is designed as ananalyst support toolfor financial research, not a real-time chatbot.
Response times of 10–30 seconds are acceptable in contexts where analysts currently spend minutes
manually searching SEC filings.
F CASESTUDY: SELF-CORRECTION INACTION
We present a detailed example illustrating how the self-correction loop recovers from an incomplete
first-pass answer.
Example: Self-Correction Recovering Incomplete Answer
Question:“What was Apple’s total revenue in FY2023 and how did it compare to FY2022?”
Attempt 1:
•Retrieval Agent: Selectshybrid filterwith entity “AAPL”, retrievesk=10documents
•Reasoning Agent: Generates “Apple’s revenue in 2023 was $394.3B.”
•Judge Agent: Score 0.4 (answer missing FY2022 comparison,incomplete)
•Decision: Score< τ 1= 0.5→RETRY
Attempt 2 (Escalated):
•Retrieval Agent: Escalates tok=20, RSE enabled for segment merging
•Reasoning Agent: “Apple’s FY2023 revenue was $383.3B, down 2.8% from $394.3B in FY2022.”
•Judge Agent: Score 0.85 (complete: both years present, comparison included)
•Decision: Score≥τ 2= 0.4→ACCEPT
This example illustrates key mechanisms: the Judge identifiessemantic incompletenessrather than
surface errors, escalated retrieval provides additional context, and threshold decay allows acceptance
of good-but-not-perfect answers after retry effort. Every decision is logged with full provenance
(question ID, retrieval parameters, judge scores, decisions, latency), enabling compliance officers to
trace any answer back to its source documents.
16

Published as a conference paper at ICLR 2026
F.1 ERRORANALYSIS
To understand system limitations, we analyze failure modes where Self-Improving RAG fails to
improve over single-pass baselines:
•Retrieval ceiling(28% of failures): When relevant information is absent from the document
corpus, escalated retrieval cannot recover
•Arithmetic errors(22%): Multi-step calculations accumulate rounding errors or apply
incorrect formulas
•Unit/scale confusion(18%): Misinterpreting units (thousands vs. millions) or mixing
absolute values with percentages
•Temporal misalignment(15%): Extracting figures from incorrect fiscal periods
•Judge miscalibration(12%): The Judge scores an incorrect answer highly, preventing
beneficial retry
•Hallucinated figures(5%): The model generates specific numerical values not present in
retrieved context
The dominance of retrieval ceiling failures suggests that expanding the document corpus or improving
chunking strategies could yield further gains. Judge miscalibration represents an opportunity for
calibration tuning.
STATEMENT ONLLM USAGE
In accordance with ICLR’s policy on Large Language Models, we declare:
Text Refinement:LLMs assisted with grammar and clarity improvements.
Citation Verification:We developed an automated citation verification agent that cross-referenced
all claims against source abstracts via Semantic Scholar to ensure citation accuracy.
Figure Design:AI tools assisted with figure design and layout.
All technical contributions, experimental results, and analysis were conceived and verified by the
authors.
17