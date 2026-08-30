# Execution-Anchored Hallucination Calibration Reranking for Verilog Code Generation

**Authors**: Guang Yang, Xing Hu, Xiang Chen, Terry Yue Zhuo, Xin Xia

**Published**: 2026-08-24 08:11:43

**PDF URL**: [https://arxiv.org/pdf/2608.22938v1](https://arxiv.org/pdf/2608.22938v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet their performance degrades significantly on low-resource Hardware Description Languages such as Verilog. While multi-candidate sampling improves the likelihood of generating correct solutions, au-tomatically selecting the optimal candidate remains an open challenge. Through a systematic empirical study across nine models and two benchmarks, we identify two critical limitations:(1) existing execution-based reranking methods, which rely on testbench pass/fail outcomes, exhibit poor domain transferability due to low-quality generated testbenches; and (2) LLM-as-a-Judge suffers from reasoning hallucination, producing incon-sistent judgments for execution-equivalent code. These findings reveal two signal types with orthogonal errors: execution signals(deterministic but testbench coverage limited)and reasoning signals (semantically rich but hallucination-prone). Their orthog-onality suggests combining the two signals, yet in our experiments letting the reasoner directly observe execution results merely anchors its judgments on test outcomes; we therefore acquire the two signals independently and fuse them only at the decision stage. Based on these insights, we propose EAHC, an Execution-Anchored Hallucination Calibration reranking framework that anchors reasoning judgments to execution behavior so that execution-equivalent candidates receive consistent scores, which implements a dual-channel architecture: EAHC-R, a 4B reasoning discriminator; and EAHC-T, a testbench generator leveraging RAG for execution verification.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 1
Execution-Anchored Hallucination Calibration
Reranking for Verilog Code Generation
Guang Yang, Xing Hu∗, Xiang Chen, Terry Yue Zhuo, and Xin Xia
Abstract—Large Language Models (LLMs) have demonstrated
remarkable capabilities in code generation, yet their performance
degrades significantly on low-resource Hardware Description
Languages such as Verilog. While multi-candidate sampling
improves the likelihood of generating correct solutions, au-
tomatically selecting the optimal candidate remains an open
challenge. Through a systematic empirical study across nine
models and two benchmarks, we identify two critical limitations:
(1) existing execution-based reranking methods, which rely on
testbench pass/fail outcomes, exhibit poor domain transferability
due to low-quality generated testbenches; and (2) LLM-as-a-
Judge suffers fromreasoning hallucination, producing incon-
sistent judgments for execution-equivalent code. These findings
reveal two signal types with orthogonal errors: execution signals
(deterministic but testbench coverage limited) and reasoning
signals (semantically rich but hallucination-prone). Their orthog-
onality suggests combining the two signals, yet in our experiments
letting the reasoner directly observe execution results merely
anchors its judgments on test outcomes; we therefore acquire
the two signals independently and fuse them only at the decision
stage. Based on these insights, we proposeEAHC, an Execution-
Anchored Hallucination Calibration reranking framework that
anchors reasoning judgments to execution behavior so that
execution-equivalent candidates receive consistent scores, which
implements a dual-channel architecture:EAHC-R, a 4B reason-
ing discriminator fine-tuned on 47K compiler-verified judgment
traces via multi-teacher distillation; andEAHC-T, a testbench
generator leveraging RAG over a 53K corpus for execution
verification. Experiments show thatEAHCranks first in 15
of 18 configurations and attains the best average on both
benchmarks, elevating average Pass@1 from 53.99% to 65.10%
on VerilogEval-v2 and from 53.18% to 68.25% on ResBench,
recovering over 60% of the gap to Pass@10 oracle.
Index Terms—Large Language Models, Verilog Code Gener-
ation, Code Reranking, LLM-as-a-Judge, Hardware Description
Languages
I. INTRODUCTION
Hardware Description Languages (HDLs), including Ver-
ilog, VHDL, and SystemVerilog, form the foundation of mod-
ern digital circuit design [1]. Among these, Verilog stands out
for its widespread industry adoption, serving as the standard
in both ASIC and FPGA design workflows. As integrated
Corresponding author: Xing Hu.
Guang Yang is with the State Key Laboratory of Blockchain and Data
Security, Zhejiang University, Hangzhou, China, and also with the Hangzhou
High-Tech Zone (Binjiang) Institute of Blockchain and Data Security,
Hangzhou, China. Xing Hu and Xin Xia are with the State Key Laboratory of
Blockchain and Data Security, Zhejiang University, Hangzhou, China. Xiang
Chen is with the School of Artificial Intelligence and Computer Science,
Nantong University, Nantong, China. Terry Yue Zhuo is with the Department
of Computer Science, Monash University and CSIRO’s Data61, Australia.
E-mail: novelyg@outlook.com, xinghu@zju.edu.cn, xchencs@ntu.edu.cn,
terry.zhuo@monash.edu, xin.xia@acm.org.
Manuscript received April 19, 2020; revised August xx, xxxx.circuit complexity continues to grow, automated Verilog code
generation has become increasingly critical to reduce devel-
opment costs and alleviate engineering workload [2]. With
the rapid advancement of Large Language Models (LLMs),
remarkable progress has been achieved in code generation
forhigh-resourceprogramming languages, with state-of-the-
art LLMs now surpassing 90% Pass@1 on Python benchmarks
like HumanEval [3], [4]. Inspired by this success, researchers
have attempted to transfer LLM capabilities to Verilog gen-
eration [5]. However, these efforts reveal a significant perfor-
mance gap: even the most capable LLMs achieve only 30–50%
Pass@1 on Verilog benchmarks, far below their performance
on general-purpose languages [6], [7].
The performance gap stems from fundamental challenges:
(1)data scarcity, as available Verilog corpora are orders of
magnitude smaller than those for general-purpose languages
and (2)domain complexity, since hardware design involves
temporal logic, parallel execution, and signal propagation that
diverge fundamentally from sequential software paradigms. A
straightforward solution is to build large-scale, high-quality
Verilog corpora. However, it is limited by the escalating
labeling cost and the high computational expense of retraining
large models for this domain.
An alternative strategy leverages the characteristics of LLM
decoding: by samplingkcandidates at non-zero temperature,
the probability thatat least oneis correct (Pass@k) far
exceeds that of a single greedy attempt (Pass@1). Table I
reveals a striking disparity: while Pass@10 reaches 71.30% on
VerilogEval-v2 [6] and 77.78% on ResBench [8], Pass@1 lags
significantly at 53.99% and 53.18%, respectively. This17–25
percentage-point gapreveals considerable latent capability
that existing approaches fail to fully leverage in practice. In
real-world development scenarios, developers typically expect
definitive code suggestions rather thankuncertain alternatives,
which require additional manual effort to evaluate and select.
This gap motivates thecode rerankingproblem: givenkcan-
didatesY k={ˆy 1, . . . ,ˆy k}for requirementx, design a scoring
function to select the most likely correct implementation,
effectively converting sampling diversity into deployment-
ready accuracy.
To better understand the limitations of existing approaches,
we conduct a systematic empirical study (Section III) across
nine code generation models and two benchmarks, evaluating
state-of-the-art reranking methods including generation proba-
bility [9], semantic matching [10], execution verification [11],
and LLM-as-a-Judge [12]. Our study reveals two critical
findings:
✩Finding 1: Poor Domain Transferability.Existing
arXiv:2608.22938v1  [cs.SE]  24 Aug 2026

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 2
reranking methods for general-purpose languages generalize
poorly to Verilog. Probability-based approaches suffer from
distributional shift between training corpora and HDL syntax,
while semantic matching methods fail to capture hardware-
specific constructs such as alwaysblocks and non-blocking
assignments. Execution-based methods like CodeT [11] har-
nessexecution signals for candidate selection, but their re-
liance on self-generated testbenchs limits effectiveness: the
underlying code models lack hardware domain expertise to
produce high-quality test cases.
✩Finding 2: Reasoning Hallucination in LLM-as-a-
Judge.In contrast to execution-based methods, LLM-as-a-
Judge leveragesreasoning signals and emerges as the most
competitive baseline, yet it exhibits a critical flaw:reasoning
hallucination, where it may produce inconsistent correctness
judgments forexecution-equivalentcode, i.e., candidates with
identical input–output behavior on a finite test suite (Sec-
tion III). Specifically, for candidatesˆy iandˆy jwith the same
execution behavior, the LLMs may give different judgments.
This inconsistency stems from the fact that LLMs reason at
thetokenlevel without grounding inexecution semantics[13].
These findings reveal two signal types with distinct strengths
and limitations. Execution-based methods (Finding 1) lever-
ageexecution signals, i.e., deterministic pass/fail outcomes
where execution-equivalent code must produce identical re-
sults. Their strength lies inconsistency: identical behavior
guarantees identical scores. However, they suffer fromcov-
erage gaps, as limited testbenches cannot detect all bugs.
LLM-as-a-Judge (Finding 2) relies onreasoning signals, i.e.,
semantic correctness judgments derived from analyzing code
logic. Their strength lies insemantic coverage: reasoning can
identify errors beyond what tests exercise. However, they
suffer fromreasoning hallucination, producing inconsistent
judgments for execution-equivalent code.
Crucially, the two limitations differ in character: execution
errors are systematic, missing the same behaviors whenever
the tests fail to exercise them, whereas reasoning errors
are stochastic, varying across implementations that behave
identically. This asymmetry is what makes the signals worth
combining: execution outcomes can stabilize fluctuating judg-
ments, while reasoning can cover what the tests leave untested.
How to combine them is less obvious. The intuitive option
is interactive: let the reasoner observe execution results before
judging. Information theory bounds what this can gain, but
only when the resulting verdict is a revision of the judgment
the reasoner would have produced anyway (Section IV-B);
the bound does not rule out every interactive design. Em-
pirically, however, exposing execution feedback anchors the
reasoner on test outcomes and costs accuracy (Section V). We
therefore keep the channels apart, a design we formalize as a
dual-channel information model (Section IV-B): the execution
channel generates testbenchs solely from requirementx, the
reasoning channel judges correctness from(x,ˆy)pairs without
observing execution results, and the two signals meet only
when the final choice is made. Our ablations support the
premise behind this design: reasoning does carry information
that execution misses (Section V).
We realize this design asEAHC(Execution-AnchoredHallucinationCalibration), a reranking framework with one
component per channel.EAHC-Rimplements the reasoning
channel: we curate a 47K dataset via multi-teacher distillation
with compiler-in-the-loop verification and fine-tune Qwen3-4B
to acquire Verilog-specific judgment capabilities; at inference,
majority voting across multiple samples yields aggregated
reasoning signals.EAHC-Timplements the execution chan-
nel: we construct a 53K dataset using the same distillation
pipeline and employ RAG techniques to produce high-quality
execution signals. The two signals meet only in hierarchical
selection, the decision stage anticipated above: candidates
sharing identical execution vectors are grouped into execution-
equivalence clusters and assigned uniform reasoning scores,
then ranked by fusing execution pass rates with cluster-level
reasoning confidence. Since equivalent implementations share
a single score, the winning cluster no longer depends on which
candidate the judge happens to favor. This is how execution
anchors reasoning, and what we mean bycalibration: consis-
tency of judgments, not conversion of scores into probabilities.
We evaluateEAHCon VerilogEval-v2 and ResBench across
nine code generation models, including commercial (GPT-
5, DeepSeek-V3, and GLM-4.6), general-purpose (Qwen2.5-
Coder, Open-Coder, and Seed-Coder), and Verilog-specialized
(HaVen, VeriPrefer, and CodeV-R1) models. Results demon-
strate that: (1)EAHCimproves average Pass@1 from 53.99%
to65.10%on VerilogEval-v2 and from 53.18% to68.25%on
ResBench, outperforming the strongest baseline by+5.91%
and+4.96%respectively, and ranking first in 15 of 18 con-
figurations; (2) ablation studies confirm both components are
essential, withEAHC-R contributing +7.12% andEAHC-T
contributing +5.75% on average; (3) the fusion weightα=0.6
performs best in our sweep, reflecting the higher reliability of
execution signals; (4)EAHCis orthogonal to training-based
approaches, providing gains of 8.93% to 19.64% across base,
SFT, and RL-optimized models; and (5) independent fusion
outperforms interactive fusion across all evaluated configura-
tions.
In summary, this paper makes the following contributions:
•Problem Formulation and Empirical Findings.We for-
malize the Verilog code reranking problem and conduct the
first systematic empirical study across nine code generation
models and two benchmarks.
•Dual-Channel Framework and Domain Resources.We
proposeEAHC, a dual-channel reranking framework that
acquires execution and reasoning signals independently and
fuses them only at the decision stage, a choice supported
by our information-theoretic analysis (Theorem 1) and by
a direct comparison against interactive fusion. It also yields
two reusable resources, VeriJudge-47K and VeriTest-53K,
curated by multi-teacher distillation with compiler-in-the-
loop verification, along with the 4B judge and testbench
generator trained on them.
•State-of-the-Art Performance.EAHCattains the best av-
erage Pass@1 on both benchmarks and the best reranking
accuracy in 15 of the 18 configurations (9 models×2
benchmarks), recovering over 60% of the gap between
Pass@1 and Pass@10.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 3
To support future research, we open-source our trained mod-
els and code1to facilitate reproducibility and future research
in Verilog code generation.
II. BACKGROUND ANDRELATEDWORK
A. Verilog Code Generation
Verilog is a hardware description language widely used for
designing digital circuits, including ASICs and FPGAs [1].
Recent work has explored leveraging LLMs for automated
Verilog generation. Early efforts fine-tuned code models on
curated Verilog corpora [7], [14], while more recent ap-
proaches incorporate reinforcement learning with compiler
feedback [15], [16] or retrieval-augmented generation [17],
[18]. Despite these advances, Verilog generation remains chal-
lenging due to the scarcity of training data and the complexity
of hardware semantics.
a) Task Formulation.:Given a natural language speci-
ficationxdescribing the desired hardware functionality, the
goal of Verilog code generation is to produce a syntactically
correct and functionally accurate Verilog moduleˆy:
ˆy=M(x;θ)(1)
whereMdenotes an LLM with parametersθ. Correctness
is evaluated by executingˆyagainst a ground-truth testbench
T∗. When samplingncandidates, the standard metric Pass@k
(k≤n) estimates the probability that at least one ofkselected
candidates is correct [3]:
Pass@k=E problems"
1− n−c
k
 n
k#
(2)
wherecdenotes the number of correct candidates amongn
samples. In practice, Pass@1 reflects single-attempt accuracy,
while Pass@k(e.g.,k=10) reveals the upper-bound potential
achievable through effective candidate selection.
B. Code Reranking
Code reranking addresses the problem of selecting the best
candidate from multiple LLM-generated solutions. Existing
approaches can be broadly categorized into four paradigms: (1)
Generation probability, which ranks candidates by their like-
lihood under the LLM [9]; (2)Semantic matching, which uses
embedding similarity between requirements and code [19];
(3)Execution verification, which leverages test case execution
as a selection signal [11], [20]; and (4)LLM-as-a-Judge,
which employs LLMs to directly assess code correctness [12],
[21]. We provide detailed descriptions of baseline methods in
Section III.
The last two paradigms have been paired before, but only in
coupled forms: a verifier consumes execution results as input
features, or a model rewrites its code after seeing test failures.
Such coupling makes the judgment a function of the execution
outcome, which is what our analysis bounds (Section IV-B)
and what our experiments find costly.EAHCinstead keeps
the two acquisitions apart, letting them meet only when the
ranking is decided.
1https://github.com/NTDXYG/EAHC CODEa) Task Formulation.:We formally define the code
reranking problem as follows. Given a natural language spec-
ificationxand a set ofkcandidate implementationsY k=
{ˆy1,ˆy2, . . . ,ˆy k}sampled from an LLM, the goal is to design
a scoring functionR:X × Y →Rsuch that the top-ranked
candidate maximizes correctness probability:
ˆy∗= arg max
ˆyi∈YkR(x,ˆy i)(3)
The objective is to maximize Pass@1 after reranking, thereby
converting the latent potential of Pass@kinto realized single-
attempt accuracy.
III. EMPIRICALSTUDY
To understand the limitations of existing reranking methods
on Verilog code generation, we conduct a systematic empirical
study.
A. Experimental Setup
Datasets.We evaluate on two established Verilog bench-
marks: (1)VerilogEval-v2[6], containing 156 problems with
human-written specifications and golden testbenchs; (2)Res-
Bench[8], comprising 56 problems focused on more complex,
realistic hardware designs. Both benchmarks provide ground-
truth testbenchs for functional verification.
Code Generation Models.To ensure comprehensive cover-
age, we select 9 representative LLMs spanning three categories
with diverse capabilities and specializations: (1)Commercial
models: GPT-5 [22], DeepSeek-V3.2 [23], and GLM-4.6 [24],
representing state-of-the-art proprietary systems with strong
general reasoning abilities; (2)General-purpose code mod-
els: Qwen2.5-Coder-7B [25], OpenCoder-8B [26], and Seed-
Coder-8B [27], which are open-source models optimized for
code generation across multiple programming languages; (3)
Verilog-specialized models: HaVen [28], VeriPrefer [29], and
CodeV-R1 [30], which are fine-tuned specifically on Verilog
corpora to enhance hardware code generation. For each model,
we samplek=10candidates per problem using temperature
τ=1.0to ensure sufficient diversity among generated solu-
tions.
Existing Reranking Methods.We evaluate representative
methods from four paradigms:
(1)Generation Probability.Probranks candidates by their
length-normalized log-probability under the generation model:
Rprob(ˆy) =1
|ˆy||ˆy|X
t=1logp θ(yt|x, y <t)(4)
CodeReviewer[9] extends this by combining forward gener-
ation probability with backward reconstruction likelihood:
Rrev(x,ˆy) = logp(ˆy|x) + logp(x|ˆy)(5)
The backward term measures how well the code can recon-
struct the original requirement, providing a mutual information
perspective.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 4
TABLE I
EMPIRICAL STUDY RESULTS ONVERILOGEVAL-V2ANDRESBENCH. BEST BASELINE RESULTS PER COLUMN AREBOLDED. “–”INDICATES
UNAVAILABLE PROBABILITY FORAPI-BASED MODELS.
Method GPT-5 DS-V3 GLM-4 QC OC SC HaVen VeriPref CodeV Avg.
VerilogEval-v2
Pass@1 85.90 73.08 76.28 33.33 31.41 47.44 40.38 41.67 56.41 53.99
Probability – – – 25.64 28.21 44.23 33.33 37.82 43.59 35.47
CodeReviewer – – – 25.64 28.85 43.59 32.69 37.82 43.59 35.36
CodeRank-Q84.6273.08 80.77 30.13 35.26 48.72 40.38 41.03 53.21 54.13
CodeRank-J 83.33 75.00 82.69 30.77 33.97 47.44 36.54 37.18 56.41 53.70
CodeT-Self 78.21 73.08 82.69 35.90 38.46 48.72 41.03 37.18 57.05 54.70
CodeT-GPT 78.21 72.4485.90 39.1039.1052.56 49.36 53.21 62.82 59.19
DiTing-1.5B 82.69 74.36 82.05 35.26 36.54 48.08 42.31 46.79 55.13 55.91
DiTing-7B 83.3376.2882.69 36.5439.7451.28 46.79 50.64 58.97 58.47
Pass@10 (Oracle) 92.31 85.26 92.95 53.21 56.41 67.31 58.33 66.67 69.23 71.30
ResBench
Pass@1 73.21 64.29 67.86 41.07 42.86 42.86 46.43 46.43 53.57 53.18
Probability – – – 30.36 33.93 35.71 53.57 42.86 42.86 39.88
CodeReviewer – – – 32.14 30.36 33.93 57.14 41.07 44.64 39.88
CodeRank-Q75.0064.29 75.00 39.29 46.43 46.43 58.93 55.36 57.14 57.54
CodeRank-J 71.43 71.43 71.43 32.14 41.07 46.43 51.79 51.79 48.21 53.97
CodeT-Self 73.21 73.21 76.79 44.64 42.86 50.00 53.57 53.5760.7158.73
CodeT-GPT 73.2176.79 80.36 57.14 53.57 60.7155.36 53.57 58.9363.29
DiTing-1.5B 69.64 67.86 67.86 48.21 46.43 55.36 60.71 50.00 58.93 58.33
DiTing-7B 71.43 64.29 71.43 48.21 48.21 46.4366.07 58.93 60.7159.52
Pass@10 (Oracle) 85.71 83.93 87.50 73.21 71.43 73.21 75.00 75.00 75.00 77.78
(2)Semantic Matching.CodeRank[19] projects require-
ments and code into a shared embedding space and ranks by
cosine similarity:
Rembed(x,ˆy) =ϕ(x)⊤ϕ(ˆy)
∥ϕ(x)∥ · ∥ϕ(ˆy)∥(6)
whereϕ(·)is a pretrained code embedding model. We evaluate
two embedding models: Qwen3-Embedding [31] and Jina-
Code-v2 [32].
(3)Execution Verification.CodeT[11] generates testbenchs
alongside candidates and uses execution agreement as the
ranking signal. Given a generated testbench setT, candidates
are clustered by execution outcomes, and cluster scores com-
bine code count with test count:
Rcodet(ˆy) =|C ˆy| × |T ˆy|(7)
whereC ˆyis the set of candidates sharing the same execution
vector asˆy, andT ˆyis the set of tests they all pass. We evaluate
two variants: CodeT-Self (testbenches generated by the same
model) and CodeT-GPT (testbenches generated by GPT-5).
(4)LLM-as-a-Judge.Code-DiTing[12] employs fine-tuned
judge models to assess code correctness. Givennindependent
judgments, the score is computed via majority voting:
Rjudge(x,ˆy) =1
nnX
j=1⊮h
F(j)
ϕ(x,ˆy) =Yesi
(8)
whereF(j)
ϕdenotes thej-th sampled judgment from the LLM
judgeF ϕ. We evaluate two model sizes: Code-DiTing-1.5B
and Code-DiTing-7B, with majority voting.Evaluation Metrics.We report Pass@1 as the primary metric,
measuring the accuracy of the top-ranked candidate. The
original Pass@1refers to the correctness of the candidate
generated via greedy decoding (temperatureτ=0), repre-
senting the default LLM output without any reranking. For
reranking methods, Pass@1 reflects the correctness of the
selected candidate after ranking theksampled alternatives. We
also report Pass@k(Oracle) as the upper bound, representing
the probability that at least one correct solution exists among
k=10samples. The gap between original Pass@1 and Ora-
cle quantifies the potential improvement achievable through
effective reranking.
Implementation Details.All experiments are conducted on
NVIDIA RTX4090 GPUs. For LLM-as-a-Judge methods, we
setn=3sampling rounds in majority voting with temperature
0.6. For CodeT, we generate 5 testbenchs per problem. Verilog
simulation is performed using Icarus Verilog.
B. Results and Analysis
Table I presents the reranking performance across all con-
figurations. We first analyze the characteristics of each method
category, then summarize two critical findings.
1) Method-wise Analysis:Generation Probability.
Probability-based methods (Prob, CodeReviewer) consistently
underperform, achieving only 35.47% and 35.36% average
Pass@1 on VerilogEval-v2, evenlowerthan the original
Pass@1 (53.99%). This degradation stems from distributional
shift: Verilog syntax differs from the natural language and

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 5
Design a 4-bit synchronous counter module 
with the following specifications:
Module Name: counter_4bit
Inputs:
- clk: Clock signal, positive edge triggered
- rst_n: Active-low asynchronous reset
Outputs:
- cnt[3:0]: 4-bit counter output
Functional Requirements:
1. The counter shall increment by 1 on each rising edge 
of the clock signal.
2. When rst_n is asserted low (logic 0), the counter 
shall reset to 0 immediately (asynchronous reset).
3. When rst_n is deasserted high (logic 1), normal 
counting operation resumes.
4. The counter shall wrap around from 4'b1111 (15) to 
4'b0000 (0) and continue counting.
Timing Requirements:
- All state transitions occur on the positive edge of clk.
- Reset is asynchronous and takes effect regardless of 
clock state.
module counter (
    input  wire clk,
    input  wire rst_n,
    output reg [3:0] cnt
);
    always @(posedge clk or negedge rst_n)
        if (~rst_n)
            cnt <= 4'd0;
        else
            cnt <= cnt + 4'd1;
endmodulemodule counter (
    input  wire clk,
    input  wire rst_n,
    output reg [3:0] cnt
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 4'b0000;
        else
            cnt <= cnt + 1'b1;
    end
endmodule
<think>
Let me analyze this counter 
implementation:
1. The module uses 
asynchronous active...
This is a well-structured 
synchronous counter with 
proper reset handling.
</think>
<answer>Yes</answer>LLM-as-a-Judge
<think>
Let me analyze this counter 
implementation:...
The implementation has 
potential timing and 
synthesis issues.
</think>
<answer>No</answer>LLM-as-a-Judge
Fig. 1. Example of reasoning hallucination
high-resource programming languages, causing unreliable
likelihood estimates. Notably, these methods are inapplicable
to API-based models where token probabilities are unavailable.
Semantic Matching.CodeRank variants show modest
improvements on some models but remain inconsistent.
On VerilogEval-v2, CodeRank-Q achieves 54.13% average
Pass@1, only marginally above the original Pass@1. The
limitation lies in embedding models’ inability to capture
Verilog-specific semantics.
Execution Verification.CodeT’s performance is heavily de-
pendent on the quality of generated testbenchs. CodeT-Self,
which relies on the same model to generate testbenchs,
achieves limited improvements (54.70% on VerilogEval-v2,
58.73% on ResBench) due to the poor quality of self-generated
test cases. CodeT-GPT leverages GPT-5 for testbench gen-
eration and achieves substantially better results (59.19% and
63.29%), emerging as the best-performing baseline on average.
However, this variant requires expensive API calls for each
problem, limiting its practical applicability.
LLM-as-a-Judge.Code-DiTing shows competitive perfor-
mance, with DiTing-7B achieving 58.47% on VerilogEval-v2
and 59.52% on ResBench. Scaling from 1.5B to 7B yields
consistent improvements (+2.5% on average), suggesting rea-
soning capability matters. However, despite being the most
stablebaseline across models, LLM-as-a-Judge exhibits a
critical flaw that limits its reliability (detailed in Finding 2).
2) Finding 1: Poor Domain Transferability:Reranking
methods designed for general-purpose languages exhibit lim-
ited effectiveness on Verilog, with most failing to substantially
outperform original Pass@1. As shown in Table I, only CodeT-
GPT and DiTing-7B achieve meaningful improvements over
the original Pass@1. On VerilogEval-v2, the best baseline
(CodeT-GPT, 59.19%) still leaves a 12.11 percentage point
gap to the oracle (71.30%), indicating thatover 70% of thepotential improvement remains unrealized. Similar patterns
emerge on ResBench, where CodeT-GPT achieves 63.29%
against an oracle of 77.78%.
This poor transferability stems from the fundamental mis-
match between method assumptions and HDL characteristics.
Probability-based methods assume reliable likelihood esti-
mates, which fail under distributional shift. Semantic matching
relies on embeddings that lack HDL-specific representations.
Execution-based methods depend on test quality, yet self-
generated testbenchs for Verilog exhibit low coverage and
frequent compilation failures.
3) Finding 2: Reasoning Hallucination:Among all base-
lines, LLM-as-a-Judge achieves the most consistent perfor-
mance. However, we identify a critical limitation:reason-
ing hallucination, i.e., the tendency to produce inconsistent
judgments for execution-equivalent code. To quantify this
phenomenon, we call two candidatesˆy iandˆy jexecution-
equivalent when they produce identical outputs on all ground-
truth test cases, i.e.,e i=ej. We then examine whether LLM
judgments respect this equivalence relationship.
They frequently do not. Figure 1 illustrates a concrete
example: two counter implementations with identical wave-
forms receive opposite verdicts. One is praised for “correct
synchronous design,” while the other is criticized for “potential
timing issues,” despite both passing all functional tests. The
root cause is that LLMs reason at thetokenlevel without
grounding inexecution semantics. Judgments become sensitive
to superficial code variations rather than actual functional
behavior. While majority voting [33] reduces random noise,
it cannot eliminate this systematic inconsistency.
IV. METHOD
To address the limitations identified in Section III, we pro-
poseEAHC, a reranking framework that leverages execution
signals to calibrate LLM reasoning, which is shown in Fig. 2.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 6
Algorithm 1:EAHCFramework
Input:CandidatesY k, requirementx, fusion weightα
Output:Selected candidateˆy∗
// Phase 1: Execution Anchoring
1T ←EAHC-T(x);// Generate testbench
2for each ˆyi∈ Ykdo
3e i←Execute(ˆy i,T);// Get execution
vector
// Phase 2: Equivalence Clustering
4{C e} ←Cluster(Y k,{ei});// Group by
execution
// Phase 3: Fusion Scoring
5for each cluster Cedo
6S exec(Ce)←PassRate(e);
7S reason(Ce)←max ˆy∈C eFϕ(x,ˆy);
8S hybrid(Ce)←α·S exec+ (1−α)·S reason ;
// Phase 4: Hierarchical Selection
9C∗←arg max CeShybrid(Ce);
10ˆy∗←arg max ˆy∈C∗Fϕ(x,ˆy);
11return ˆy∗
A. Framework Overview
Given a set ofkcandidate implementationsY k=
{ˆy1, . . . ,ˆy k}for requirementx,EAHCselects the optimal
candidate through a two-stage process:
ˆy∗= arg max
ˆy∈C∗Fϕ(x,ˆy),whereC∗= arg max
CeShybrid(Ce)
(9)
whereC edenotes a execution-equivalence cluster based on
execution vectore,S hybrid is the fusion score combining
execution and reasoning signals, andF ϕis the reasoning
discriminator.
Algorithm 1 summarizes the overall workflow: (1)Exe-
cution Anchoring: generate testbenchs and obtain execution
vectors for all candidates; (2)Equivalence Clustering: group
candidates by execution vectors; (3)Fusion Scoring: compute
hybrid scores for each cluster; (4)Hierarchical Selection:
select the best cluster, then the best individual within it.
B. Dual-Channel Information Model
This subsection asks two questions: what each channel adds
to the other, and how the two should be combined.
1) Channel Independence:We formalize correctness judg-
ment as adual-channel information acquisition problem, in
which each channel forms its signal without observing the
other’s:
•Execution Channel→Execution SignalE: Generates
testbenchs solely from requirementx, without observing
candidate codeˆy. The resulting execution outcomes (pass/-
fail on each test) constitute theexecution signal.
•Reasoning Channel→Reasoning SignalR: Judges cor-
rectness from(x,ˆy)pairs through semantic analysis, without
observing execution results. The judgment (Yes/No with
reasoning chain) constitutes thereasoning signal.
Separating acquisition preserves the orthogonality of the
two error sources: execution errors aresystematic, followingfrom limited coverage, whereas reasoning errors arestochastic,
following from hallucination.
2) Signal Complementarity:To reason about what the two
signals jointly reveal, we adopt the approximationP(E, R|
Y)≈P(E|Y)·P(R|Y). Because both channels read the
same requirementx, this is a modeling choice rather than a
property we can guarantee (Section VI-B).
Proposition 1(Mutual Information Complementarity).By the
chain rule, the joint use ofEandRsatisfies
I(Y;E, R) =I(Y;E)+I(Y;R|E)≥max{I(Y;E), I(Y;R)}
(10)
with a strict gain iffI(Y;R|E)>0.
The identity holds unconditionally, and the approximation
above is instead what licenses the additive fusion score below.
What neither settles is whetherI(Y;R|E)>0, that is,
whether reasoning stays informative once execution is known.
This is an empirical question, and our ablations answer it
affirmatively (Section V).
3) Independent Fusion vs. Interactive Fusion:A natural
alternative isinteractive fusion, where the reasoner observes
execution results before judging. We analyze its post-hoc form,
in which an acquiredRis revised in light ofE; a judge that
keepsRintact while separately exploitingElies outside the
analysis.
Theorem 1(Non-Superiority of Post-Hoc Interactive Fusion).
LetRbe acquired independently of execution, and letR′=
h(R,E). Then
I(Y;R′|E)≤I(Y;R|E),(11)
with equality whenR′retains allY-relevant information inR
givenE.
Proof.SinceR′is computed from(R,E), the variables form
a Markov chainY→(R,E)→R′, and the Data Processing
Inequality [34] givesI(Y;R′,E)≤I(Y;R,E). Subtracting
I(Y;E)from both sides yields the claim.
Equivalently,I(Y;E, R′)≤I(Y;E, R): the pair an inter-
active design ends up with is never more informative than the
pair we keep. The degenerate case is instructive: a reasoner that
merely echoes execution,R′=f(E), satisfiesI(Y;R′|E)=0
and contributes nothing.EAHC-R therefore judgeswithout
execution feedback. We read this as a reason to acquire the
signals independently, not as a proof that independence is
optimal: the result bounds information content rather than the
accuracy of any particular scoring rule, andEAHCfuses the
two signals linearly rather than optimally. Section V-C reports
what the interactive design actually costs.
4) Posterior Fusion:Assuming the conditional-
independence approximation, Bayes’ theorem yields the
additive form
logP(Y=1|E, R)∝logP(E|Y=1)|{z }
Sexec+ logP(R|Y=1)|{z }
Sreason(12)
with reliability weightα:
Shybrid =α·S exec+ (1−α)·S reason (13)

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 7
Fig. 2.EAHCFramework
System Prompt for Code Verification
Please serve as a Verilog code verification expert. Your task
is to analyze the provided Verilog code and determine if it
correctly implements the problem requirements.
Code Verification Task:
1)Code Analysis Phase: Conduct a thorough logic and
functional analysis of the provided Verilog code.
2)Test Case Generation Phase: Generate comprehensive
test cases covering expected functionality, edge cases, and
potential failure modes.
3)Verification Phase: Evaluate whether the code will cor-
rectly pass all generated test cases.
Output Format:Provide only “Yes” or “No” within the
<answer></answer>tags.
Fig. 3. Domain prompt template for reasoning generation inEAHC-R.
In practice we instantiateS execandS reason as bounded surro-
gates for these log-likelihoods, a test pass rate and a judgment
frequency (Sections IV-D and IV-C); the derivation thus moti-
vates the additive form rather than yielding an exact posterior.
C. EAHC-R: Reasoning Discriminator
As illustrated in Fig. 2 (Part II, top branch),EAHC-R
functions as an LLM-as-a-Judge module that estimates the
functional correctness probabilityF ϕ(x,ˆy)∈[0,1]for each
candidate.
1) Independence Constraint:To preserve channel indepen-
dence,EAHC-R receivesonlythe requirementxand candi-
date codeˆyas input,excludingany execution information
(testbench, execution results, pass rates). The reasoner judges
correctness through pure semantic analysis of code logic.
2) Training Data Construction:As shown in Fig. 2 (Part
I, top pipeline), we construct the VeriJudge-47K high-quality
(x,ˆy,reasoning,label)dataset through multi-teacher knowl-
edge distillation:
1)Candidate Generation: We use Seed-Coder to sample
diverse Verilog candidates from existing datasets with
ground-truth testbenches [35].2)Reasoning Generation: For each candidate, we apply
domain prompt engineering to elicit reasoning chains from
two teacher models: DeepSeek-R1 and GLM-4. We de-
liberately employ multiple teachers to increase reason-
ing diversity and reduce single-model bias. The struc-
tured outputs contain<think>...</think>reasoning
and<answer>Yes/No</answer>verdicts. Fig. 3 illus-
trates our verification prompt template.
3)Compiler-in-the-loop Verification: We execute candidates
against ground-truth testbenches using Icarus Verilog. Only
samples where the LLM’s verdict aligns with execution
results are retained, effectively filtering hallucinated rea-
soning.
3) Model Training and Inference:We fine-tune Qwen3-
4B on the curated VeriJudge-47K dataset using LoRA with
PiSSA [36] initialization. The lightweight 4B model enables
single-GPU deployment while reducing high-latency inference
cost compared to commercial API calls. At inference, we
samplen=3reasoning chains per candidate and aggregate via
majority voting to obtain thereasoning signal:
Fϕ(x,ˆy) =1
nnX
j=1⊮h
F(j)
ϕ(x,ˆy) =Yesi
(14)
whereF(j)
ϕdenotes thej-th sampled reasoning chain.F ϕ
is thus an endorsement frequency rather than a probability
of correctness, and reranking depends only on the order it
induces. V oting removes variance across samples for a single
candidate; consistency across candidates is enforced later by
execution anchoring (Section IV-E).
D. EAHC-T: Testbench Generator
EAHC-T generates high-quality testbenches to obtain exe-
cution signals for anchoring (see Fig. 2, bottom-left for data
construction and right side for workflow).
1) Independence Constraint:As illustrated in the workflow
(Fig. 2, right),EAHC-T receivesonlythe natural language
requirement as input,excludingcandidate implementations.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 8
RAG Prompt for Testbench Generation
System:You are a Verilog testbench generation expert.
User:Please search the knowledge base for relevant test-
benches and then generate the testbench.
Problem:[requirementx]
Module Header:[interfaceh]
Assistant:Here are the retrieved testbenches from the
knowledge base:
[Top-Kretrieved examplesR K]
User:Please refer to the above knowledge base and generate
fiveVerilog testbench cases. Do not implement the module,
only generate the testbench.
[One-shot example with expected format]
Assistant:...
Fig. 4. RAG prompt template for testbench generation inEAHC-T.
This design ensures that the same testbench fairly evaluates
allkcandidates and that the execution signalEis acquired
without access to the reasoning signalR.
2) VeriTest-53K: Testbench Knowledge Distillation:As
shown in Fig. 2 (bottom-left), we construct the VeriTest-53K
high-quality(x, h,testbenches)dataset through the following
pipeline:
1)Seed Data Collection: We collect NL-Verilog pairs from
PyraNet [37] as seed examples, deliberately using a differ-
ent data source from VeriJudge-47K. This separation keeps
the two channels from being adapted on the same corpus,
which limits one avenue for correlated behavior without by
itself establishing conditional independence.
2)Domain Prompt Engineering: Using GPT-4o and GLM-
4.6 as teachers, we generate 5 testbench cases for each
sample through carefully designed prompts. Employing
multiple teachers improves test coverage diversity, as differ-
ent LLMs tend to generate test cases focusing on different
aspects.
3)Compiler-in-the-loop Verification: Generated testbenches
are validated using Icarus Verilog. Only samples that com-
pile successfully and follow correct format are retained.
Both corpora are screened against the evaluation bench-
marks; Section VI-D quantifies the residual overlap.
3) RAG-Enhanced Generation:At inference, we employ
a two-stage retrieval-augmented generation strategy, as illus-
trated in Fig. 4.
Stage 1: Similarity Retrieval.Given requirementxand
module interfaceh, we use BM25 to retrieve top-Ksimilar
examples from VeriTest-53K:
RK=Top-K
(q,t)∈K[BM25(x⊕h, q)](15)
Stage 2: Context-Augmented Generation.Inject retrieved
examples as in-context demonstrations:
T=G T(x, h| R K)(16)
whereG Treuses theEAHC-R tuned 4B model (Section IV-C)
without additional fine-tuning, relying on RAG withK=5
retrieved examples for domain adaptation.4) Execution Anchoring:As shown in Fig. 2 (right, EAHC-
T block), each candidateˆy iis executed against the generated
testbenchTcontainingmtest cases. The execution vector
ei∈ {−1,0,1}mrecords the outcome:
eij=

1ifˆy ipasses testt j
0ifˆy ifails testt j
−1if compilation fails(17)
Candidates with identical execution vectors (e i=ej) form
anexecution-equivalencecluster; sinceTis generated rather
than exhaustive, a cluster may still mix implementations that
differ only where the tests are silent, which is exactly where
the reasoning channel is needed. The execution score for a
clusterC eis computed as the test pass rate:
Sexec(Ce) =Pm
j=1⊮[ej= 1]Pm
j=1⊮[ej̸=−1](18)
E. Hierarchical Selection
The final stage combines execution and reasoning signals
through hierarchical selection (see Fig. 2, right side).
1) Equivalence Clustering:Candidates are grouped into
execution-equivalence clusters based on their execution vec-
tors:
Ce={ˆy i∈ Yk|ei=e}(19)
As illustrated in Fig. 2, candidates with identical execution
behavior (e.g., Verilog 1, 3 in clusterC 1; Verilog 2, 7 in cluster
C2) are grouped together.
2) Fusion Scoring:For each clusterC e, we compute the
hybrid score by fusing execution and reasoning signals:
Shybrid(Ce) =α·S exec(Ce) + (1−α)·S reason(Ce)(20)
The reasoning score aggregates individual judgments using
max:
Sreason(Ce) = max
ˆyi∈CeFϕ(x,ˆy i)(21)
which represents the most optimistic reasoning estimate within
the cluster. Taking the maximum lets one confident judgment
carry a cluster, which is in tension with the consistency that
anchoring aims for. We setα= 0.6by default, giving slightly
higher weight to the more reliable execution signal.
3) Two-Level Selection:Level 1: Cluster Selection.Select
the cluster with highest hybrid score:
C∗= arg max
CeShybrid(Ce)(22)
For tie-breaking, we apply the priority order:S hybrid> S exec>
Sreason>|C|.
Level 2: Individual Selection.Within the optimal cluster
C∗, select the candidate with highest reasoning score:
ˆy∗= arg max
ˆyi∈C∗Fϕ(x,ˆy i)(23)
The final outputˆy∗thus balances execution reliability with
reasoning-based quality assessment.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 9
TABLE II
RQ1: EFFECTIVENESS EVALUATION RESULTS ONVERILOGEVAL-V2ANDRESBENCH. THE BEST RERANKING RESULT PER COLUMN ISBOLDED;
PASS@1AND THEPASS@10ORACLE ARE LISTED FOR REFERENCE.
Method GPT-5 DS-V3 GLM-4 QC OC SC HaVen VeriPref CodeV Avg.
VerilogEval-v2
Pass@1 85.90 73.08 76.28 33.33 31.41 47.44 40.38 41.67 56.41 53.99
CodeT-Self 78.21 73.08 82.69 35.90 38.46 48.72 41.03 37.18 57.05 54.70
CodeT-GPT 78.21 72.4485.9039.10 39.10 52.56 49.36 53.21 62.82 59.19
DiTing-1.5B 82.69 74.36 82.05 35.26 36.54 48.08 42.31 46.79 55.13 55.91
DiTing-7B 83.33 76.28 82.69 36.54 39.74 51.28 46.79 50.64 58.97 58.47
EAHC-T (k=1) 83.97 76.28 80.13 35.90 37.82 48.72 41.03 38.46 55.77 55.34
EAHC-T (k=3) 83.97 76.92 82.05 39.10 41.03 50.64 44.87 45.51 57.69 57.98
EAHC-R (n=1) 83.97 76.92 79.49 39.74 46.79 61.54 50.00 56.41 61.54 61.82
EAHC-R (n=3) 82.05 78.85 80.13 41.67 47.44 60.90 52.56 57.69 63.64 62.77
EAHC 83.97 81.4181.4146.15 50.00 62.18 53.85 60.26 66.67 65.10
Pass@10 (Oracle) 92.31 85.26 92.95 53.21 56.41 67.31 58.33 66.67 69.23 71.30
ResBench
Pass@1 73.21 64.29 67.86 41.07 42.86 42.86 46.43 46.43 53.57 53.18
CodeT-Self 73.21 73.21 76.79 44.64 42.86 50.00 53.57 53.57 60.71 58.73
CodeT-GPT 73.2176.79 80.3657.14 53.57 60.71 55.36 53.57 58.93 63.29
DiTing-1.5B 69.64 67.86 67.86 48.21 46.43 55.36 60.71 50.00 58.93 58.33
DiTing-7B 71.43 64.29 71.43 48.21 48.21 46.43 66.07 58.93 60.71 59.52
EAHC-T (k=1) 73.21 67.86 67.86 46.43 46.43 46.43 58.93 53.57 57.14 57.54
EAHC-T (k=3) 75.00 66.07 71.43 58.93 55.36 58.93 64.29 57.14 55.36 62.50
EAHC-R (n=1) 71.43 73.21 69.64 55.36 55.36 57.14 64.29 57.14 60.71 62.70
EAHC-R (n=3) 71.43 66.07 76.79 51.79 53.57 55.36 69.64 60.71 62.50 63.10
EAHC 76.7973.21 78.5760.71 60.71 67.86 71.43 60.71 64.29 68.25
Pass@10 (Oracle) 85.71 83.93 87.50 73.21 71.43 73.21 75.00 75.00 75.00 77.78
V. EVALUATION
We evaluateEAHCthrough four research questions:
•RQ1 (Overall Effectiveness): How effective isEAHC
compared to existing code reranking methods on Verilog
generation tasks?
Motivation: Our empirical study (Section III) revealed that
existing methods suffer from poor domain transferability and
reasoning hallucination. We evaluate whetherEAHC’s dual-
channel design effectively addresses these limitations.
•RQ2 (Orthogonality): IsEAHCorthogonal and comple-
mentary to training-based optimization methods (SFT and
RL)?
Motivation: Training-based methods (SFT, RL) have been
widely adopted to improve code generation models. Since
these methods optimize the model’s generation distribution
whileEAHCoptimizes candidate selection, we investigate
whether the two optimization dimensions are orthogonal
and can be combined for additional gains across different
training stages.
•RQ3 (Independent vs. Interactive Fusion): Does inde-
pendent channel fusion outperform interactive fusion in
practice?
Motivation: Our analysis bounds what a reasoner gains
from observing execution results, but leaves open how a
concrete implementation behaves. We therefore compare
independent fusion against the alternative where the reasonersees execution feedback before judging.
•RQ4 (Validity of the Execution Anchor): How reliable
are the testbenches thatEAHCgenerates, and how does the
framework behave when they carry no information?
Motivation: Clustering and scoring both depend on self-
generated testbenches, so the execution channel must be
validated rather than assumed. We measure it against the
hidden ground-truth testbenches and examine the problems
where it fails to separate candidates.
A. RQ1: Overall Effectiveness
1) Performance Comparison:Table II presents the experi-
mental results on VerilogEval-v2 and ResBench.EAHCattains
the highest average Pass@1 on both benchmarks and the best
reranking accuracy in 15 of the 18 configurations.
Results on VerilogEval-v2.EAHCachieves an average
Pass@1 of65.10%, outperforming the strongest baseline
CodeT-GPT (59.19%) by+5.91%absolute improvement.
Compared to the original Pass@1 (53.99%),EAHCrecovers
64.2%of the gap to the Pass@10 oracle (71.30%). The
improvements are particularly pronounced on domain-specific
models with weaker base performance: Open-Coder improves
from 31.41% to 50.00% (+18.59%), and Seed-Coder from
47.44% to 62.18% (+14.74%).
Results on ResBench.EAHCachieves an average
Pass@1 of68.25%, surpassing CodeT-GPT (63.29%) by

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 10
(a) Impact ofαon VerilogEval-v2
 (b) Impact ofαon ResBench
Fig. 5. Sensitivity analysis of hyperparameterαacross different LLMs.
+4.96%. This represents61.3%recovery of the Pass@1-
to-Pass@10 gap. Notable improvements include Qwen-Coder
(41.07%→60.71%,+19.64%) and HaVen (46.43%→71.43%,
+25.00%), demonstratingEAHC’s effectiveness on challeng-
ing generation tasks.
Analysis of Exceptions.EAHCis not the best reranker in
three configurations, and CodeT-GPT wins all of them: GLM-
4 on VerilogEval-v2 (81.41 vs. 85.90), and DS-V3 and GLM-4
on ResBench (73.21 vs. 76.79 and 78.57 vs. 80.36). The three
share a profile: a strong generator whose candidates mostly
compile and pass, evaluated against CodeT-GPT’s GPT-5-
authored testbenches. When almost every candidate is already
correct, the ranking turns on the two error modes that remain.
The first lies in the execution channel, where a permissive
testbench admits an incorrect implementation into a passing
cluster: Section V-D reports false-accept rates of 12.8% and
6.6% and mixed-cluster rates of 23.8% and 27.5%. On both
quantities the GPT-5 testbenches are stronger on ResBench
(5.9% and 25.7%), which accounts for the two exceptions
there. The second lies in the reasoning channel, where an
incorrect candidate outranks every correct one on 5.7% and
8.4% of the problems that contain one (Table III); anchoring
reduces this drift without eliminating it, which explains the
remaining exception on VerilogEval-v2.
Pairwise Significance TestingMcNemar’s test [38] is
appropriate for paired binary outcomes, comparing the number
of problems where one method succeeds and the other fails.
Letn 01denote problems where the baseline succeeds but
EAHCfails, andn 10denote the reverse. The test statistic is:
χ2=(|n01−n10| −1)2
n01+n10(24)
Compared to the best baseline CodeT-GPT,EAHC
shows highly significant improvements on both datasets. On
VerilogEval-v2, the p-value across all models is1.1×10−10,
and on ResBench, it is1.8×10−8. These results (p <0.01)
indicate that the aggregate advantage over CodeT-GPT is
unlikely to arise by chance, although, as noted above, CodeT-
GPT remains ahead on three individual configurations.2) Ablation Study:We conduct ablation studies to under-
stand the contribution of each component, as shown in Table II.
Execution Channel (EAHC-T).Using only execution-
based testcase scoring (EAHC-T,k=3) achieves 57.98% on
VerilogEval-v2 and 62.50% on ResBench. The performance
gap to fullEAHC(7.12% and 5.75%) confirms that reason-
ing signals provide substantial complementary value beyond
execution feedback.
Reasoning Channel (EAHC-R).Using only reasoning-
based hierarchical ranking (EAHC-R,n=3) achieves 62.77%
on VerilogEval-v2 and 63.10% on ResBench. The perfor-
mance degradation (2.33% and 5.15% vs. fullEAHC) demon-
strates that execution anchoring provides crucial grounding for
reasoning-based selection.
Hierarchical Selection.ComparingEAHC-R (n=1) with
EAHC-R (n=3), hierarchical selection improves performance
from 61.82% to 62.77% on VerilogEval-v2 and from 62.70%
to 63.10% on ResBench. This validates our cluster-based
design for mitigating reasoning hallucination through diversity
preservation.
Fusion Weightα.Figure 5 illustrates the sensitivity of
EAHCto the fusion weightα∈[0,1]. Performance con-
sistently peaks in the rangeα∈[0.1,0.6], with optimal
values aroundα= 0.3–0.6. Both extreme configurations
(α= 0: reasoning only;α= 1: execution only) yield
suboptimal results, with performance degrading sharply when
α >0.8. This confirms that the dual-channel fusion effectively
leverages complementary information from both sources.
3) The Inconsistency that Execution Anchoring Targets:
hierarchical selection rests on the premise thatEAHC-R scores
drift across candidates with identical behavior. We verify this
on candidates of the same problem that pass every ground-
truth test: they are execution-equivalent, so a consistent judge
should score them identically. Table III shows otherwise.
Scores disagree in 15.3% and 27.4% of equivalent pairs, and
since ranking depends on scores rather than binary verdicts,
these are the operative rates. Majority voting (n=3) reduces
within-candidate variance, lowering verdict flips from 7.5%

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 11
TABLE III
CONSISTENCY OFEAHC-R (n=3)ON EXECUTION-EQUIVALENT
CANDIDATES,AGGREGATED OVER THE NINE GENERATORS. APAIR IS TWO
CANDIDATES OF ONE PROBLEM THAT PASS ALL GROUND-TRUTH TESTS;
ONLY PAIRS WITH FULLY PARSEABLE JUDGMENTS ARE COUNTED.
Measure VerilogEval-v2 ResBench
Problems with≥2 equivalent candidates (%) 65.2 71.6
Equivalent pairs 30,502 10,040
Opposite verdicts (%) 4.8 7.1
Unequal scores (%) 15.3 27.4
Identical code, unequal scores (%) 5.2 16.4
Wrong candidate outranks all correct (%) 5.7 8.4
to 4.8% and from 12.8% to 7.1%, but cross-candidate dis-
agreement persists. Even byte-identical code receives different
scores in 5.2% and 16.4% of pairs, where no semantic ambigu-
ity can explain the gap. The cost is direct: on 5.7% and 8.4%
of problems that contain a correct candidate, an incorrect one
ranks highest, a loss voting cannot fix. Execution anchoring
eliminates this drift by assigning one score per cluster, yielding
+2.33% and +5.15% overEAHC-R. Inconsistency also rises
with generator strength (3.8% on Seed-Coder vs. 7.5% on
GPT-5 and GLM-4), consistent with the narrowing margins
in RQ1.
Summary of RQ1
EAHCachieves the highest average Pass@1 on both
benchmarks (65.10% and 68.25%, i.e. +5.91% and
+4.96% over the best baseline) and ranks first in 15 of
18 configurations, with CodeT-GPT still ahead on three
strong-generator configurations. Both execution and
reasoning channels contribute positively, and their fu-
sion performs best overall. Even after majority voting,
reasoning scores disagree on 4.8%–7.1% of execution-
equivalent pairs, the inconsistency execution anchoring
absorbs.
B. RQ2: Orthogonality
1) Motivation:Training-based methods, including super-
vised fine-tuning (SFT) and reinforcement learning (RL), have
been widely adopted to improve code generation models. A
natural question arises:IsEAHCcomplementary to training-
based approaches, or do they overlap in their improvements?
We investigate whetherEAHCcan provide consistent gains
across different stages of the model training pipeline.
2) Experimental Design:We select the CodeV model fam-
ily as our subject, which follows a typical three-stage training
pipeline for domain-specific code generation:
•Base Model (Qwen2.5-Coder): A general-purpose code
LLM pre-trained on multi-language code corpora, serving
as the foundation without any Verilog-specific optimization.
•SFT Model (CodeV-R1): Built upon Qwen2.5-Coder, this
model is supervised fine-tuned on curated Verilog code
generation datasets to acquire domain-specific knowledge
and coding patterns.
(a) Results on VerilogEval-v2
(b) Results on ResBench
Fig. 6. Orthogonality with Training-based Methods.
•RL Model (CodeV-R1-RL): Starting from CodeV-R1, this
model is further optimized using the DAPO algorithm [39]
with execution feedback as reward signals.
This progression (Base→SFT→RL) represents a common
paradigm in building domain-specific code generation models,
where each stage incrementally improves the model’s genera-
tion capability. For each model variant, we evaluate Pass@1,
Pass@1 withEAHC-T, Pass@1 withEAHC-R, Pass@1 with
fullEAHC, and Pass@10 (Oracle upper bound) to analyze
whetherEAHC’s improvements are consistent across different
training stages.
3) Results Analysis:Figure 6 presents the orthogonality
analysis across three training stages.
Consistent Improvements Across All Training Stages.
EAHCdelivers substantial improvements regardless of the
underlying model’s training stage. On VerilogEval-v2,EAHC
improves Pass@1 by+12.82%for Base (33.33%→46.15%),
+10.26%for SFT (56.41%→66.67%), and+8.98%for RL
(69.23%→78.21%). On ResBench, the corresponding im-
provements are+19.64%,+10.72%, and+8.93%. These con-
sistent gains confirm that generation optimization (via training)
and selection optimization (viaEAHC) operate on orthogonal
dimensions.
Complementary Effects.Training and reranking address
fundamentally different aspects of code generation. Training
(SFT→RL) progressively improves base Pass@1 from 33.33%
to 69.23% on VerilogEval-v2 by enhancing candidate qual-

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 12
ity. Meanwhile,EAHCrecovers a significant portion of the
Pass@1-to-Pass@10 gap at each stage: 64.5% (Base), 80.0%
(SFT), and 56.0% (RL). Notably, even the RL-optimized
model, which already incorporates execution feedback dur-
ing training, still benefits substantially fromEAHC’s selec-
tion strategy, indicating that generation-time optimization and
inference-time selection capture complementary signals.
Summary of RQ2
Within this model family,EAHCadds +8.93%–
+19.64% at every training stage, indicating that gen-
eration optimization and selection optimization ad-
dress different aspects of code generation and can be
combined. Whether the same holds for other training
pipelines remains to be tested.
C. RQ3: Independent vs. Interactive Fusion
1) Motivation:An intuitive alternative to our design is
interactive fusion, where the reasoner observes execution re-
sults before judging. Theorem 1 bounds what such interaction
can gain when it amounts to revising an acquired judgment,
but the bound admits equality and does not predict accuracy
loss. Whether interactive fusion actually hurts is therefore
empirical; we compare the two strategies across all nine
models.
2) Experimental Design:We compare two fusion strategies
under identical experimental settings:
•Interactive Fusion:EAHC-R receives execution results
(pass/fail status, error messages) as additional context before
making judgments. The reasoning and execution signals are
fusedinteractivelywithin the reasoner.
•Independent Fusion:EAHC-R judges correctness without
observing any execution information. Signals are fused
independentlyat the decision stage viaS hybrid =α·S exec+
(1−α)·S reason .
Both use the sameEAHC-R checkpoint and differ only in
the prompt; the comparison thus isolates prompt-level interac-
tion rather than end-to-end training with execution context.
3) Results:Table IV presents the comparison. The results
support our design choice:independent fusion outperforms
interactive fusion on every model and benchmark, with
average gains of+4.49%on VerilogEval-v2 and+4.56%on
ResBench.
Performance Degradation Pattern.The gap is smallest
for the strongest models (GPT-5 +1.28, GLM-4 +1.92 on
VerilogEval-v2) and largest for the weaker ones (up to +7.70
for VeriPrefer). Interactive fusion thus appears most harmful
when reasoning must discriminate among lower-quality can-
didates.
Anchoring Bias Analysis.Qualitative inspection ofEAHC-
R’s reasoning chains reveals pronouncedanchoring biasunder
interactive fusion:
•Pass Anchoring: When all tests pass, the reasoner tends to
approve the candidate even when subtle logical errors exist
that the testbench fails to cover.•Fail Anchoring: When tests fail, the reasoner over-penalizes
the candidate even when the failure stems from testbench
limitations rather than code errors.
Anchoring is one instance of the post-hoc revision that
Theorem 1 covers: the reasoning channel drifts toward echoing
execution outcomes instead of contributing independently. The
accuracy loss is consistent with the lossy end of that bound
rather than with equality; whether a judge trained to distrust
unreliable feedback could approach equality remains open.
Comparison with Execution-Only.Notably, interactive
fusion (60.61% / 63.69%) performs only marginally better than
execution-onlyEAHC-T (57.98% / 62.50%), suggesting that
the reasoning signal’s contribution is largely “absorbed” by
execution information. In contrast, independent fusion retains
far more of it, achieving 65.10% / 68.25%.
Summary of RQ3
Independent fusion outperforms interactive fusion by
+4.49%on VerilogEval-v2 and+4.56%on ResBench,
confirming the advantage of independent acquisition
under prompt-level interaction. Exposing execution
feedback induces anchoring bias that reduces reason-
ing’s independent contribution, causing it to degenerate
toward execution-only performance.
D. RQ4: Validity of the Execution Anchor
1) Motivation:Clustering and fusion rest on testbenches
thatEAHCwrites for itself, so the execution channel must be
validated rather than trusted as an oracle. We ask how reliable
it is and how the framework behaves where it is not.
2) Experimental Design:Each benchmark hides a ground-
truth testbench used only for measurement, never exposed to
the reranker. Running it on the ten candidates per problem
gives a reference label; running the generated testbench gives
the signalEAHC-T actually consumes. Comparing the two
yields afalse-acceptrate (passed by the generated testbench
but rejected by the reference), afalse-rejectrate (the converse),
the fraction of clusters mixing correct and incorrect candidates,
and the AUC of the execution score against the reference label.
We also measure the GPT-5 testbenches used by CodeT-GPT
as a non-retrieval reference point.
We evaluate by fault detection rather than structural cov-
erage: coverage measures how much of asingleimplemen-
tation is exercised, whereas the anchor must separateseveral
implementations of one specification. A testbench with loose
expected values can achieve full coverage on a correct module
yet return the same verdict for all candidates. False acceptance
is a fault detection rate over the incorrect implementations the
nine generators actually produced.
3) Results:The anchor is informative but conservative.
Atk=5, false acceptance is 12.8% and 6.6% while false
rejection reaches 47.3% and 49.6% (Table V): the anchor
rarely certifies an incorrect candidate but often fails to certify
a correct one (AUC 0.743 and 0.727). This asymmetry is
whyEAHCfuses the score rather than filters on it, since
discarding every rejected candidate would remove about half

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 13
TABLE IV
INDEPENDENT FUSION VS.INTERACTIVE FUSION ACROSS ALL MODELS. BEST RESULTS PER COLUMN AREBOLDED.
Strategy GPT-5 DeepSeek-V3 GLM-4 Qwen-Coder Open-Coder Seed-Coder HaVen VeriPrefer CodeV Avg.
VerilogEval-v2
Interactive Fusion 82.69 78.21 79.49 41.67 44.87 56.41 48.08 52.56 61.54 60.61
Independent Fusion83.97 81.41 81.41 46.15 50.00 62.18 53.85 60.26 66.67 65.10
∆+1.28 +3.20 +1.92 +4.48 +5.13 +5.77 +5.77 +7.70 +5.13+4.49
ResBench
Interactive Fusion 75.00 69.64 75.00 55.36 55.36 62.50 66.07 55.36 58.93 63.69
Independent Fusion76.79 73.21 78.57 60.71 60.71 67.86 71.43 60.71 64.29 68.25
∆+1.79 +3.57 +3.57 +5.35 +5.35 +5.36 +5.36 +5.35 +5.36+4.56
TABLE V
QUALITY OF THE EXECUTION ANCHOR AGAINST THE HIDDEN
GROUND-TRUTH TESTBENCHES,AGGREGATED OVER THE NINE
GENERATORS. LOWER IS BETTER FORF-ACCEPT, F-REJECT,ANDMIXED.
Testbench source Compiles F-accept F-reject Mixed AUC
VerilogEval-v2
GPT-5 (CodeT-GPT) 92.9 15.4 40.0 20.2 0.768
EAHC-T (k=1) 38.4 8.5 69.0 30.3 0.618
EAHC-T (k=5) 82.3 12.8 47.3 23.8 0.743
ResBench
GPT-5 (CodeT-GPT) 58.9 5.9 59.2 25.7 0.702
EAHC-T (k=1) 64.3 8.0 73.7 39.4 0.571
EAHC-T (k=5) 75.0 6.6 49.6 27.5 0.727
of the correct ones. Roughly a quarter of the clusters still
mix correct and incorrect candidates, which reasoning must
separate.
Retrieval depth and testbench source.Atk=1 only 38.4%
of testbenches compile on VerilogEval-v2 and AUC falls to
0.618, which is why the main experiments usek=5. Against
the GPT-5 testbenches, retrieval improves every measure on
ResBench but not on VerilogEval-v2, where GPT-5 compiles
more often and separates candidates slightly better. Anchor
quality thus varies with the source, and the fusion weightα
absorbs this variance.
Behavior under an uninformative anchor.We split the
1,872 VerilogEval-v2 instances by whether the generated test-
bench separates the ten candidates. On the 1,117 instances
where it returns one verdict for all,EAHCreaches 79.43%
vs. 78.51% forEAHC-R and 74.92% forEAHC-T: with
no execution signal, the ranking follows reasoning rather
than collapsing. On the remaining 755 it reaches 43.74% vs.
40.31% and 33.79%, so the anchor contributes where it carries
information. The first group holds the easier problems (oracle
82.81% vs. 50.07%), reflecting the absence of anything to
separate rather than a testbench limitation alone.
Summary of RQ4
The anchor false-accepts 6.6%–12.8% but false-rejects
47.3%–49.6%, favoring fusion over filtering. When
it separates no candidates,EAHCtracks reasoning
(79.43% vs. 78.51%) instead of degrading; when it
does separate, it gains most (43.74% vs. 40.31%).VI. DISCUSSION
A. Design Choices and Channel Independence
A core design principle ofEAHCis the independent ac-
quisition of execution and reasoning signals. To achieve this,
we employ an asymmetric architecture: LoRA fine-tuning for
the reasoning channel (EAHC-R) and Retrieval-Augmented
Generation (RAG) for the execution channel (EAHC-T).
Justification for Asymmetric Design.Table VI shows that
both channels need the tuned backbone, but they acquire
domain knowledge differently. ForEAHC-R, LoRA fine-
tuning on VeriJudge-47K is highly effective for internalizing
judgment capabilities, whereas adding RAG to an untuned
model falls short.EAHC-T reuses that same judgment-tuned
adapter and obtains its testbench knowledge from retrieval
rather than from weights: an untuned model compiles poorly
even with RAG, yet training on VeriTest-53K without retrieval
still trails retrieving from it (60.12 vs. 62.50 on ResBench).
Test structures and assertion formats are thus better supplied
as in-context examples than absorbed into parameters, which
is what makes the architecture asymmetric.
TABLE VI
ABLATION ON FINE-TUNING ANDRAGCONFIGURATIONS FOREAHC-R
ANDEAHC-T. BEST RESULTS AREBOLDED.
Configuration VerilogEval-v2 ResBench
Reasoning Channel (EAHC-R)
Untuned Base Model 56.12 57.13
Untuned + RAG 59.34 60.07
Tuned on VeriJudge-47K (Current) 62.77 63.10
Execution Channel (EAHC-T)
Untuned + RAG 55.77 58.33
Tuned on VeriJudge-47K (No RAG) 55.34 57.54
Tuned on VeriJudge & VeriTest (No RAG) 56.84 60.12
Tuned on VeriJudge-47K + RAG (Current) 57.98 62.50
Preserving Channel Independence.A natural concern
is whether sharing one base model (and LoRA adapter)
introduces coupling. At inference,EAHC-R sees only the
requirement and candidate code whileEAHC-T sees only the
requirement and retrieved examples, so no execution outcome
reaches the reasoner. A shared backbone does not rule out
statistical dependence, however: both channels can inherit the
same misconception and fail together. We count this among
the sources of correlated error discussed in Section VI-B.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 14
B. Theoretical Assumptions and Boundaries
The analysis in Section IV-B explains why independent ac-
quisition is a sound default; it does not show that independent
fusion beats every interactive design. We discuss below where
its assumptions can break.
Conditional Independence and Its Violation.Proposi-
tion 1 usesP(E, R|Y)≈P(E|Y)·P(R|Y).
Since both channels read the same requirementx, an under-
constrained specification can make them fail together: the
testbench generator verifies one misreading while the reasoner
endorses it. Our ablations show that substantial non-redundant
information survives on these benchmarks, but they do not
establish exact independence.
We measure the violation directly. On instances where at
least one candidate is correct and one is not, both channels
select an incorrect candidate on 16.1% of them, 1.71×the
9.4% independence predicts (Table VII); the ratio is 1.73 on
ResBench. Dependence tracks design style: sequential logic
roughly doubles per-channel error and more than doubles
joint failure, with state machines the worst case, because such
specifications commonly leave reset polarity or unenumerated-
state behavior implicit. Joint failure is also inflated by prob-
lems that are simply hard for both channels, so these rates
bound correlated error rather than isolate a shared misreading;
multi-clock designs, absent from both benchmarks, remain an
unquantified risk.
TABLE VII
FAILURE OF EACH CHANNEL ONVERILOGEVAL-V2,OVER THE533
INSTANCES CONTAINING BOTH A CORRECT AND AN INCORRECT
CANDIDATE. UNDER INDEPENDENCEP(BOTH)WOULD EQUAL
P(T)P(R),WHICH IS0.094OVERALL.
Stratumn P(T)P(R)P(both)
All 533 0.381 0.248 0.161
Combinational 254 0.244 0.185 0.094
Sequential 279 0.505 0.305 0.222
Finite-state machine 155 0.484 0.374 0.258
Markov Assumption and Theorem Scope.Theorem 1
applies the DPI to post-hoc interactionR′=h(R,E), which
presumes a Markov chain. With parameters fixed, the chain
holds; it would break if the judge had memorized the reference
implementation, sinceR′could then carry information about
Ythat neitherRnorEsupplies, motivating our contamination
screening (Section VI-D). The scope is also narrower than
interaction in general: it says nothing about a judge that retains
its independent assessment while exploitingEseparately. Our
preference for independent acquisition therefore rests on the
RQ3 measurements rather than a general optimality claim.
Faithfulness of Mutual Information.I(Y;R)quantifies
verdict utility: a correct Yes/No may follow a flawed chain,
so the measure reflects the verdict’s value rather than the
soundness of the reasoning. Compiler-in-the-loop filtering and
majority voting reduce verdict noise but neither inspects the
chain. Our claims are correspondingly about verdict-level
information.Does the Training Filter Collapse Reasoning into Exe-
cution?VeriJudge-47K retains only candidates whose teacher
verdict agrees with execution (Section IV-C). The two targets
differ: the filter uses ground-truth testbenches, so the retained
label tracksY, whereasEat inference comes from a generated
five-case testbench with coverage gaps (Section III). Training
towardYthus does not reduce to imitatingE; accordingly,
EAHC-R alone outperformsEAHC-T alone (62.77 vs. 57.98
on VerilogEval-v2), which an approximation ofEcould not
achieve. The filter does cost coverage of hard instances:
candidates that every teacher misjudged are discarded, leaving
the judge weakest where judgment is hardest.
Cluster Reasoning Aggregation.Each cluster inherits the
maxof its members’ reasoning scores. Becausemaxprop-
agates the most confident judgment, one hallucinatedYes
can promote a cluster on its own. Table VIII comparesmax
against mean and median:maxstill ranks first, by under
one point. Averaging suppresses an isolated hallucination but
also penalizes a correct cluster with uneven endorsements;
the second effect dominates here. The aggregator choice is in
any case second-order: clustering itself recovers +2.33% and
+5.15% overEAHC-R (Section V-A3), an order of magnitude
more than separatesmaxfrom median.
TABLE VIII
CLUSTER REASONING AGGREGATION ABLATION. AVERAGEPASS@1
OVER THE NINE GENERATORS.
Aggregation VerilogEval-v2 ResBench
max(default)65.10 68.25
Mean 64.60 67.66
Median 64.32 67.26
C. Broader Implications and HDL Specificity
Importance to Software Engineering.Verilog defects prop-
agate into physical fabrication, making them far costlier to
fix than software defects, which can still be patched after
release. As LLM-based Verilog generation gains industrial
adoption without principled quality assurance, the unique
challenges of HDLs (e.g., parallel semantics, timing) present
critical opportunities for SE techniques like test generation and
reranking [40]–[42].
Are the Limitations Exclusive to HDLs?While poor domain
transferability and reasoning hallucination exist in general-
purpose languages, they aremarkedlymore severe in Verilog.
For example, execution-based methods yield substantial gains
on Python [11] but struggle on Verilog due to LLMs’ inabil-
ity to generate high-quality hardware testbenches. Combined
with data scarcity and complex semantics, these exacerbated
challenges motivate targeted frameworks likeEAHC.
D. Contamination and Diversity Analysis
Both benchmarks release only a natural-language prompt,
a module header, and a hidden testbench; neither publishes a
reference implementation. Overlap can therefore be measured
only on the problem side, so we represent each benchmark
item by its prompt and header, which is also the query

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 15
EAHC-T issues to the retrieval corpus at inference; VeriTest-
53K questions state the same two fields and are thus di-
rectly comparable. Each problem is scored by its nearest
corpus neighbour under word-level ROUGE-L and LFM2.5-
Embedding-350M cosine similarity.
No benchmark problem has a neighbour above 0.9 on either
measure (Table IX); the closest reach 0.667 ROUGE-L and
0.862 cosine. Median lexical overlap is 0.285 and 0.437,
whereas cosine exceeds 0.7 for most pairs: both sides are
Verilog specifications sharing vocabulary and framing, which
is domain relatedness rather than problem-level leakage.
TABLE IX
NEAREST-NEIGHBOUR OVERLAP BETWEEN BENCHMARK PROBLEMS
(PROMPT+HEADER)AND THEVERITEST-53KRETRIEVAL CORPUS.
ROUGE-L Embedding cosine
Benchmarkmed. p90 max≥0.9 med. p90 max≥0.9
VerilogEval-v2 (156) 0.285 0.400 0.667 0% 0.750 0.800 0.862 0%
ResBench (56) 0.437 0.621 0.764 0% 0.730 0.813 0.850 0%
Exact port signatures (direction, bit width, and name, ig-
noring clock/reset-only stubs) recur for 52/156 VerilogEval-
v2 problems but only 3/56 in ResBench. Such matches reflect
standard HDL skeletons rather than shared problems: an 8-
bit input to 8-bit output interface constrains nothing about the
function to implement, and the more distinctive interfaces in
ResBench almost never recur.
Diversity is the other side. VeriJudge-47K holds 47,375
records over 12,195 distinct problem statements (each paired
with multiple candidates); VeriTest-53K pairs one testbench
with each of its 53,015 distinct questions, so exact repetitions
are already collapsed. Scoring every distinct statement by its
nearestotherstatement in the same corpus gives a median
cosine of 0.945 and 0.897, seemingly high but expected in
a domain where every item is a Verilog specification. These
values calibrate Table IX: the closest benchmark problem
reaches only 0.862, below the typical intra-corpus similarity.
Benchmark problems thus lie further from our corpora than
corpus items lie from each other.
Residual overlap is benign: a training record pairs a re-
quirement with a sampled candidate and a Yes/No verdict, so a
recurring requirement carries no reference answer. Pre-training
exposure remains outside our control, but every reranker in
Section V scores the same candidates and therefore shares it.
E. Threats to Validity
Internal Validity.To avoid the risk of implementation errors,
we use unit testing and manual verification on sampled cases.
To reduce randomness from candidate sampling and majority
voting, we fix random seeds across all experiments. To miti-
gate data contamination, we remove samples with ROUGE-
L>0.9against any benchmark problem from VeriJudge-
47K and VeriTest-53K, and audit the residual overlap in
Section VI-D.
External Validity.We evaluate on VerilogEval-v2 (156 prob-
lems) and ResBench (56 problems), which may not fully
represent industrial-scale designs. Future work should validate
EAHCon larger proprietary benchmarks. We focus on Verilog;while the dual-channel design is language-agnostic in princi-
ple, transferring it to another HDL requires rebuilding both
corpora and retraining, so we make no claim beyond Verilog.
Section V-A instead characterizes where the method under-
performs within its stated scope. Additionally, the VeriJudge-
47K and VeriTest-53K datasets are curated using LLMs (GPT-
4o and GLM-4), which may introduce teacher model biases.
We mitigate this by using multi-teacher distillation and strict
compiler-in-the-loop verification.
Construct Validity.Correctness is defined by execution on
ground-truth testbenches, which may miss subtle bugs. Sec-
tion V-D finds the generated testbenches permissive, leaving
about a quarter of clusters mixed. The fusion weightα=0.6
is a global constant that cannot adapt to an individual weak
testbench; performance is stable overα∈[0.1,0.6](Fig. 5),
and conditioningαon per-problem anchor quality is left to
future work.
Cost and Practicality.On a single RTX 4090, per-problem
latency is∼15–20s(vs.∼2s greedy), a∼10×increase
dominated by testbench generation (∼1K tokens), simulation
(∼0.5s), and reasoning calls (∼30K tokens total). Cheaper
settings hurt:n=1leaves contradictory verdicts at 7.5%/12.8%
instead of 4.8%/7.1% (Section V-A3), andk=1compiles only
38.4% of testbenches (Section V-D). Our configuration is thus
near the diminishing-returns point; remaining latency is better
addressed by confidence-based early stopping, quantization,
or a smaller distilled judge, which we leave to future work.
Whether the cost is justified depends on the generator: gains
reach +18.59 on Open-Coder while the strongest models leave
little for any reranker to recover, so the overhead is best spent
where the Pass@1-to-Pass@10 gap is wide.
VII. CONCLUSION
We address the Verilog code reranking problem to bridge the
gap between Pass@kpotential and Pass@1 reality. Through
empirical analysis, we identify two critical limitations: poor
domain transferability and reasoning hallucination. We pro-
poseEAHC, a dual-channel framework that independently ac-
quires execution and reasoning signals and anchors reasoning
to execution behavior so that execution-equivalent candidates
are scored alike. It raises average Pass@1 by over 11% and
15% on VerilogEval-v2 and ResBench, ranking first in 15
of the 18 configurations. Future work includes extending to
other low-resource HDLs, validating on industrial-scale bench-
marks, and exploring formal verification as a complementary
signal.
ACKNOWLEDGEMENTS
This work was supported by National Key R&D Program of
China (No. 2024YFB4506400). Guang Yang is also supported
by the Postdoctoral Fellowship Program of CPSF under Grant
Number GZC20260902.
REFERENCES
[1] P. Flake, P. Moorby, S. Golson, A. Salz, and S. Davidmann, “Verilog
hdl and its ancestors and descendants,” Proceedings oftheACM on
Programming Languages, vol. 4, no. HOPL, pp. 1–90, 2020.

IEEE TRANSACTIONS ON SOFTWARE ENGINEERING, VOL. XX, NO. XX, XX 2026 16
[2] G. Yang, W. Zheng, X. Chen, D. Liang, P. Hu, Y . Yang, S. Peng,
Z. Li, J. Feng, X. Wei etal., “Large language model for verilog
code generation: Literature review and the road ahead,” arXiv preprint
arXiv:2512.00020, 2025.
[3] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan,
H. Edwards, Y . Burda, N. Joseph, G. Brockman, A. Ray, R. Puri,
G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan,
S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian,
C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis,
E. Barnes, A. Herbert-V oss, W. H. Guss, A. Nichol, A. Paino, N. Tezak,
J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse,
A. N. Carr, J. Leike, J. Achiam, V . Misra, E. Morikawa, A. Radford,
M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder,
B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba,
“Evaluating large language models trained on code,” 2021. [Online].
Available: https://arxiv.org/abs/2107.03374
[4] X. Du, M. Liu, K. Wang, H. Wang, J. Liu, Y . Chen, J. Feng, C. Sha,
X. Peng, and Y . Lou, “Evaluating large language models in class-level
code generation,” in Proceedings oftheIEEE/ACM 46th International
Conference onSoftware Engineering, 2024, pp. 1–13.
[5] S. Joel, J. Wu, and F. Fard, “A survey on llm-based code generation
for low-resource and domain-specific programming languages,” ACM
Transactions onSoftware Engineering andMethodology, 2024.
[6] M. Liu, N. Pinckney, B. Khailany, and H. Ren, “Verilogeval: Evaluating
large language models for verilog code generation,” in 2023 IEEE/ACM
International Conference onComputer Aided Design (ICCAD). IEEE,
2023, pp. 1–8.
[7] S. Thakur, B. Ahmad, H. Pearce, B. Tan, B. Dolan-Gavitt, R. Karri,
and S. Garg, “Verigen: A large language model for verilog code
generation,” ACM Transactions onDesign Automation ofElectronic
Systems, vol. 29, no. 3, pp. 1–31, 2024.
[8] C. Guo and T. Zhao, “Resbench: A resource-aware benchmark for
llm-generated fpga designs,” in Proceedings ofthe15th International
Symposium onHighly Efficient Accelerators and Reconfigurable
Technologies, 2025, pp. 25–34.
[9] T. Zhang, T. Yu, T. Hashimoto, M. Lewis, W.-T. Yih, D. Fried,
and S. Wang, “Coder reviewer reranking for code generation,”
inProceedings ofthe 40th International Conference onMachine
Learning, ser. Proceedings of Machine Learning Research, A. Krause,
E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett,
Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 41 832–41 846. [Online].
Available: https://proceedings.mlr.press/v202/zhang23av.html
[10] J. P. Inala, C. Wang, M. Yang, A. Codas, M. Encarnaci ´on, S. Lahiri,
M. Musuvathi, and J. Gao, “Fault-aware neural code rankers,” Advances
inNeural Information Processing Systems, vol. 35, pp. 13 419–13 432,
2022.
[11] B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou, and
W. Chen, “Codet: Code generation with generated tests,” in TheEleventh
International Conference onLearning Representations.
[12] G. Yang, Y . Zhou, X. Chen, W. Zheng, X. Hu, X. Zhou, D. Lo,
and T. Chen, “Code-diting: A reasoning-based metric for functional
alignment in code evaluation,” arXiv preprint arXiv:2505.19502, 2025.
[13] S. Wang, X. Hu, J. Chen, Z. Pan, and X. Xia, “Open the oyster:
Empirical evaluation and improvement of code reasoning confidence
in llms,” arXiv preprint arXiv:2511.02197, 2025.
[14] S. Liu, W. Fang, Y . Lu, J. Wang, Q. Zhang, H. Zhang, and Z. Xie, “Rtl-
coder: Fully open-source and efficient llm-assisted rtl code generation
technique,” IEEE Transactions onComputer-Aided Design ofIntegrated
Circuits andSystems, 2024.
[15] N. Wang, B. Yao, J. Zhou, Y . Hu, X. Wang, Z. Jiang, and N. Guan,
“Large language model for verilog generation with code-structure-
guided reinforcement learning,” in 2025 IEEE International Conference
onLLM-Aided Design (ICLAD). IEEE, 2025, pp. 164–170.
[16] Y . Wang, G. Sun, W. Ye, G. Qu, and A. Li, “Verireason: Reinforce-
ment learning with testbench feedback for reasoning-enhanced verilog
generation,” arXiv preprint arXiv:2505.11849, 2025.
[17] U. Z. Ahmed, Z. Fan, J. Yi, O. I. Al-Bataineh, and A. Roychoud-
hury, “Verifix: Verified repair of programming assignments,” ACM
Transactions onSoftware Engineering and Methodology (TOSEM),
vol. 31, no. 4, pp. 1–31, 2022.
[18] H. Qi, Y . Du, L. Zhang, S. C. Liew, K. Chen, and Y . Du, “Verirag:
A retrieval-augmented framework for automated rtl testability repair,”
arXiv preprint arXiv:2507.15664, 2025.
[19] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek,
Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy etal., “Text and code em-
beddings by contrastive pre-training,” arXiv preprint arXiv:2201.10005,
2022.[20] F. Shi, D. Fried, M. Ghazvininejad, L. Zettlemoyer, and S. I. Wang,
“Natural language to code translation with execution,” in Proceedings
ofthe2022 Conference onEmpirical Methods inNatural Language
Processing, 2022, pp. 3533–3546.
[21] Z. Zhao, R. Qiu, C. Lin, G. L. Zhang, B. Li, and U. Schlichtmann,
“Vrank: Enhancing verilog code generation from large language models
via self-consistency,” in 2025 26th International Symposium onQuality
Electronic Design (ISQED). IEEE, 2025, pp. 1–7.
[22] OpenAI, “GPT-5,” https://openai.com/gpt-5/, 2025.
[23] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng,
C. Zhang, C. Ruan etal., “Deepseek-v3 technical report,” arXiv preprint
arXiv:2412.19437, 2024.
[24] “GLM-4.6,” https://z.ai/blog/glm-4.6, 2025.
[25] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang,
B. Yu, K. Lu etal., “Qwen2. 5-coder technical report,” arXiv preprint
arXiv:2409.12186, 2024.
[26] S. Huang, T. Cheng, J. K. Liu, W. Xu, J. Hao, L. Song, Y . Xu, J. Yang,
J. Liu, C. Zhang etal., “Opencoder: The open cookbook for top-tier code
large language models,” in Proceedings ofthe63rd Annual Meeting of
theAssociation forComputational Linguistics (V olume 1:Long Papers),
2025, pp. 33 167–33 193.
[27] B. Seed, Y . Zhang, J. Su, Y . Sun, C. Xi, X. Xiao, S. Zheng, A. Zhang,
K. Liu, D. Zan etal., “Seed-coder: Let the code model curate data for
itself,” arXiv preprint arXiv:2506.03524, 2025.
[28] Y . Yang, F. Teng, P. Liu, M. Qi, C. Lv, J. Li, X. Zhang, and Z. He,
“Haven: Hallucination-mitigated llm for verilog code generation aligned
with hdl engineers,” in 2025 Design, Automation &Test inEurope
Conference (DATE). IEEE, 2025, pp. 1–7.
[29] N. Wang, B. Yao, J. Zhou, Y . Hu, X. Wang, N. Guan, and
Z. Jiang, “Insights from verification: Training a verilog generation llm
with reinforcement learning with testbench feedback,” arXiv preprint
arXiv:2504.15804, 2025.
[30] Y . Zhu, D. Huang, H. Lyu, X. Zhang, C. Li, W. Shi, Y . Wu, J. Mu,
J. Wang, P. Jin etal., “Qimeng-codev-r1: Reasoning-enhanced ver-
ilog generation,” in The Thirty-ninth Annual Conference onNeural
Information Processing Systems, 2025.
[31] Y . Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang,
D. Liu, J. Lin etal., “Qwen3 embedding: Advancing text embedding and
reranking through foundation models,” arXiv preprint arXiv:2506.05176,
2025.
[32] D. Kryvosheieva, S. Sturua, M. G ¨unther, S. Martens, and H. Xiao, “Ef-
ficient code embeddings from code generation models,” arXiv preprint
arXiv:2508.21290, 2025.
[33] L. Chen, J. Davis, B. Hanin, P. Bailis, I. Stoica, M. Zaharia, and
J. Zou, “Are more llm calls all you need? towards the scaling properties
of compound ai systems,” Advances inNeural Information Processing
Systems, vol. 37, pp. 45 767–45 790, 2024.
[34] N. J. Beaudry and R. Renner, “An intuitive proof of the data processing
inequality,” arXiv preprint arXiv:1107.0740, 2011.
[35] A. Wei, H. Tan, T. Suresh, D. Mendoza, T. S. Teixeira, K. Wang,
C. Trippel, and A. Aiken, “Vericoder: Enhancing llm-based rtl code
generation through functional correctness validation,” arXiv preprint
arXiv:2504.15659, 2025.
[36] F. Meng, Z. Wang, and M. Zhang, “Pissa: Principal singular values
and singular vectors adaptation of large language models,” Advances in
Neural Information Processing Systems, vol. 37, pp. 121 038–121 072,
2024.
[37] B. Nadimi, G. O. Boutaib, and H. Zheng, “Pyranet: A multi-layered
hierarchical dataset for verilog,” in 2025 62nd ACM/IEEE Design
Automation Conference (DAC). IEEE, 2025, pp. 1–7.
[38] Q. McNemar, “Note on the sampling error of the difference between
correlated proportions or percentages,” Psychometrika, vol. 12, no. 2,
pp. 153–157, 1947.
[39] Q. Yu, Z. Zhang, R. Zhu, Y . Yuan, X. Zuo, Y . Yue, W. Dai, T. Fan,
G. Liu, L. Liu etal., “Dapo: An open-source llm reinforcement learning
system at scale,” arXiv preprint arXiv:2503.14476, 2025.
[40] Q. Chen, N. Zhang, J. Wang, T. Tan, C. Xu, X. Ma, and Y . Li, “The
essence of verilog: A tractable and tested operational semantics for
verilog,” Proceedings oftheACM onProgramming Languages, vol. 7,
no. OOPSLA2, pp. 234–263, 2023.
[41] Z. Fang, R. Chen, Z. Yang, Y . Guo, H. Dai, and L. Wang, “Lintllm: An
open-source verilog linting framework based on large language models,”
inProceedings oftheGreat Lakes Symposium onVLSI 2025, 2025, pp.
673–680.
[42] Q. Chen, N. Zhang, J. Wang, J. Cui, T. Tan, X. Ma, C. Xu, J. Lu, and
Y . Li, “Qihe: A general-purpose static analysis framework for verilog,”
arXiv preprint arXiv:2601.11408, 2026.