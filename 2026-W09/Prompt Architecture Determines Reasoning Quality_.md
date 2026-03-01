# Prompt Architecture Determines Reasoning Quality: A Variable Isolation Study on the Car Wash Problem

**Authors**: Heejin Jo

**Published**: 2026-02-25 11:40:15

**PDF URL**: [https://arxiv.org/pdf/2602.21814v1](https://arxiv.org/pdf/2602.21814v1)

## Abstract
Large language models consistently fail the "car wash problem," a viral reasoning benchmark requiring implicit physical constraint inference. We present a variable isolation study (n=20 per condition, 6 conditions, 120 total trials) examining which prompt architecture layers in a production system enable correct reasoning. Using Claude 3.5 Sonnet with controlled hyperparameters (temperature 0.7, top_p 1.0), we find that the STAR (Situation-Task-Action-Result) reasoning framework alone raises accuracy from 0% to 85% (p=0.001, Fisher's exact test, odds ratio 13.22). Adding user profile context via vector database retrieval provides a further 10 percentage point gain, while RAG context contributes an additional 5 percentage points, achieving 100% accuracy in the full-stack condition. These results suggest that structured reasoning scaffolds -- specifically, forced goal articulation before inference -- matter substantially more than context injection for implicit constraint reasoning tasks.

## Full Text


<!-- PDF content starts -->

Prompt Architecture Determines Reasoning Quality:
A Variable Isolation Study on the Car Wash Problem
Heejin Jo
Independent Researcher
info@birth2death.com
https://www.heejinjo.me
February 25, 2026
Abstract
The car wash problem asks a simple question: “I want to wash my car. The car wash is 100
meters away. Should I walk or drive?” Every major LLM tested—Claude, GPT-4, Gemini—
recommended walking. The correct answer is to drive, because the car itself must be at the car
wash.
We ran a variable isolation study to determine which prompt architectural layer resolves
this failure. Six conditions were tested, 20 trials each, on Claude Sonnet 4.5. A bare prompt
with no system instructions scored 0%. Adding a role definition alone also scored 0%. A STAR
reasoning framework (Situation, Task, Action, Result) reached 85%. User profile injection with
physical context—car model, location, parking status—reached only 30%. STAR combined with
profile injection reached 95%. The full stack combining all layers scored 100%.
The central finding is that structured reasoning outperformed direct context injection by a
factor of 2.83×(Fisher’s exact test,p= 0.001). STAR forces the model to articulate the task
goal before generating a conclusion, which surfaces the implicit physical constraint that context
injection leaves buried. The addition of a sixth condition resolved a confound in the original
five-condition design by isolating per-layer contributions: STAR accounts for +85pp, profile
adds +10pp, and RAG provides the final +5pp to reach perfect reliability.
1 Introduction
The question originated on Mastodon, posted by Kevin (@knowmadd)1: “I want to wash my car.
The car wash is 50 meters away. Should I walk or drive?” He tested Perplexity, ChatGPT, Claude,
and Mistral. All four said walk. The correct answer is drive—you cannot wash a car that is not
there.
The post reached Hacker News and accumulated 1,499 points and 943 comments.2Discussion
centeredonthreethemes: thatLLMscannotinferimplicitprerequisiteshumanstakeforgranted; the
classical frame problem [McCarthy and Hayes, 1969], where models fail to identify which unstated
facts matter; and whether the gap between natural and structured communication is something
models can be made to bridge. Ryan Allen later published a benchmark repository3to measure this
failure across models.
1https://mastodon.world/@knowmadd/116072773118828295
2https://news.ycombinator.com/item?id=47031580
3https://github.com/ryan-allen/car-wash-evals
1arXiv:2602.21814v1  [cs.AI]  25 Feb 2026

We encountered this problem through InterviewMate, a real-time interview coaching system.
During a routine test session, the system answered “drive” while every standalone LLM we tested
said “walk.” We did not expect this. InterviewMate’s system prompt has multiple layers—role
definition, a STAR reasoning framework, user profile data, and RAG context retrieval—and we had
no way to tell which layer produced the correct answer. The result was interesting, but we could
not explain it, and a result you cannot explain is not one you can build on.
So we designed a variable isolation experiment. Instead of asking why LLMs fail at this
problem—a question the Hacker News thread had already covered extensively—we asked which
specific prompt layer fixes it within a single model.
This question has practical weight. InterviewMate operates during live interviews. The system
must interpret the interviewer’s question, retrieve relevant context from the user’s stored profile and
STAR stories, and deliver a structured answer in real time. The prompt architecture follows this
sequence:
Role definition→STAR reasoning framework→User profile (vector DB)→RAG context
retrieval
We needed to know where the reasoning quality actually comes from in this stack. The car wash
problem gave us a clean instrument for testing it: one correct answer, implicit constraint reasoning
required, and simple enough to isolate variables without confounds.
2 Related Work
TheCarWashBenchmark.RyanAllenpublishedaformalevaluationrepository(ryan-allen/car-
wash-evals) that measures car wash problem failure rates across models. His work established a
baseline: most frontier LLMs fail on first pass. Our study starts from this baseline but asks a
different question—not which models fail, but which prompt layers fix the failure within a single
model.
The Frame Problem.McCarthy and Hayes [1969] described the frame problem in classical
AI: a system must determine which facts remain unchanged when an action occurs, but has no
principled method for knowing which unstated facts are relevant. The car wash problem is a clean
modernexample. Thecar’slocationisnevermentionedinthequestion. Ahumaninfersimmediately
that the car is at home. The model does not.
Chain-of-Thought Prompting.Wei et al. [2022] showed that prompting models to reason
step-by-step significantly improves performance on multi-step tasks. Our STAR result—85% versus
a 0% baseline—is broadly consistent with this. Structured reasoning forces the model to work
through intermediate steps rather than jumping to a conclusion.
Self-Consistency.Wangetal.[2023]extendedchain-of-thoughtbysamplingmultiplereasoning
paths and selecting the most frequent answer. Our experimental design differs—we measure pass
rates across 20 independent trials per condition rather than sampling multiple paths within a single
trial—but the underlying observation is related. Reasoning quality varies across runs even with
identical prompts, and aggregation can reveal the underlying success rate.
ReAct.Yao et al. [2023a] interleaved reasoning steps with action steps, allowing models to
gather external information during the reasoning process. STAR is structurally simpler: it does not
involve external actions. But both frameworks share the principle that forcing explicit intermediate
steps before a conclusion changes the output distribution.
Tree of Thoughts.Yao et al. [2023b] proposed exploring multiple reasoning branches and
evaluating them before committing to an answer. We did not implement branching in our study,
2

but the Tree of Thoughts result supports a broader point: the structure imposed on a model’s
reasoning process has measurable effects on accuracy, independent of the information available.
Where STAR Fits.STAR (Situation, Task, Action, Result) is not a research contribution
of this paper. It is a standard interview preparation framework that we repurposed as a prompt
structure. What makes it interesting in this context is the Task step specifically. Chain-of-thought
asksthemodeltoreasonstepbystep. STARforcesthemodeltonamewhatitistryingtoaccomplish
before it begins reasoning about how. The distinction is between general sequential reasoning and
explicit goal articulation. Section 5.1 examines this mechanism in detail.
3 Methodology
This is a pilot study. The sample size (n= 20per condition, 120 total API calls) is sufficient to
identify behavioral patterns and directional differences between conditions. We report pass rates,
observed patterns, and statistical significance testing where applicable.
3.1 Experimental Setup
All trials usedclaude-sonnet-4-5-20250929via the Anthropic Python SDK. This version was
chosen for consistency with the ryan-allen/car-wash-evals baseline. Hyperparameters were fixed:
max_tokensat 512, temperature at 0.7. We chose 0.7 rather than 0 because deterministic decoding
would collapse results toward binary outcomes, making per-layer contributions impossible to dis-
tinguish. The variance introduced by temperature 0.7 across 20 runs lets us measure probabilistic
pass rates.
The prompt was modified from the original Mastodon question (50 meters) to 100 meters: “I
want to wash my car. The car wash is 100 meters away. Should I walk or drive?” This follows the
ryan-allen/car-wash-evals benchmark. The increased distance makes the surface-level heuristic (“it’s
close, just walk”) more tempting, which means correct answers require stronger implicit reasoning
to override it.
For any trial that failed or produced an ambiguous result, a challenge prompt was given: “How
will I get my car washed if I am walking?” This measures self-correction capability, which we report
as Recovery Rate.
3.2 Variable Isolation Design
Six conditions were tested. Each ran 20 independent trials.
•Condition A (Bare):No system prompt. Pure baseline to measure the model’s default
behavior.
•Condition B (Role Only):An expert advisor persona was injected as the system prompt.
•Condition C (Role + STAR):The STAR reasoning framework was added on top of the
role, requiring the model to articulate Situation, Task, Action, and Result in sequence.
•Condition D (Role + Profile):Instead of STAR, physical user context was injected—name,
location, vehicle model, current situation. Conditions C and D branch from B in parallel. They
are not cumulative.
•Condition E (Full Stack):All layers combined—Role, STAR, Profile, and simulated RAG
context.
3

•Condition F (Role + STAR + Profile):STAR and profile combined, without RAG
context. Addedto resolve the E_full_stack confound identified duringinitialanalysis, isolating
per-layer contributions between profile and RAG.
The design isolates STAR (C vs B), profile injection (D vs B), profile’s marginal contribution
on top of STAR (F vs C), RAG’s marginal contribution (E vs F), and the full interaction (E vs C,
D, and F).
3.3 Scoring Methodology
Thefirstevaluationrunusedbarewordmatching,checkingfor“walk” or“drive” intheresponse. This
failed. Every response that discussed both options was classified as ambiguous, which produced 0%
pass rates across all conditions regardless of actual recommendation. The scorer was not measuring
intent.
Run 2 replaced word matching with intent-based pattern matching: 14 pass patterns detecting
drive recommendations (e.g.,\bshould\s+drive\b) and 9 fail patterns detecting walk recommen-
dations (e.g.,\brecommend\s+walking\b).
Oneimplementationdetail: markdownboldmarkuphadtobestrippedbeforematching. Claude
frequently writes “shouldwalk” and the asterisks break whitespace-based regex patterns. When
both pass and fail patterns matched in a single response, a dominance ratio determined the result.
A 2:1 threshold was required for a definitive score; otherwise the trial was flagged ambiguous.
Statistical significance testing was performed usingscipy.stats.fisher_exact(SciPy 1.12, two-
tailed).
4 Results
4.1 Primary Pass Rates
Condition Components Pass Rate Recovery Med. Latency
A_bare No system prompt 0% (0/20) 95% 4,649ms
B_role_only Role only 0% (0/20) 100% 7,550ms
C_role_star Role + STAR 85% (17/20) 67% 7,851ms
D_role_profile Role + Profile 30% (6/20) 100% 8,837ms
F_role_star_profile Role + STAR + Profile 95% (19/20) 0% (0/1) 9,056ms
E_full_stack All combined 100% (20/20) n/a 8,347ms
Table 1: Pass rates across six experimental conditions. Pass means the first response recommends
driving. Recovery means the model self-corrects after the challenge prompt.
4.2 Key Findings
Finding 1: Reasoning structure outperforms context injection by 2.83×.C_role_star
reached85%. D_role_profilereached30%. Bothbranchfromthesamebaseline(B,0%). TheSTAR
framework forces the model to name the task before generating a conclusion. Profile injection gives
the model physical facts—car model, location, parking status—but does not force it to process
those facts in any particular order. The model can receive all the right information and still take a
shortcut past it. The difference between C (85%) and D (30%) was statistically significant (Fisher’s
4

exact test, two-tailed,p= 0.001, odds ratio= 13.22), confirming that the observed 2.83×advantage
of structured reasoning over context injection is unlikely to be attributable to chance even at this
sample size.
Finding 2: Per-layer contributions are now isolated.The addition of condition F re-
solves the confound in the original five-condition design. The progression from C to E can now be
decomposed:
•STAR alone: +85pp (0% to 85%)
•Profile on top of STAR: +10pp (85% to 95%)
•RAG on top of STAR + Profile: +5pp (95% to 100%)
Profile’s marginal contribution (+10pp) is twice that of RAG (+5pp). Both are necessary for
perfect reliability, but STAR accounts for the overwhelming majority of the improvement.
Finding 3: The recovery paradox.C_role_star had the highest first-pass accuracy but the
lowest recovery rate (67%). Conditions A, B, and D all recovered at 95–100%. F_role_star_profile
showed an even more extreme version of this pattern: its single failure (1/20) did not recover at all
(0% recovery rate). Section 5.3 discusses the mechanism.
Finding 4: Role definition alone does nothing.B_role_only scored 0%, identical to
A_bare. The expert advisor persona made the model write longer responses (median latency
7,550ms vs 4,649ms) but did not change the answer.
4.3 Failure Mode Taxonomy
Three failure patterns appeared across all failed primary trials:
Type 1—Distance Heuristic(∼70% of failures). The model treats the question as a distance
optimization problem. “100 meters is a 1–2 minute walk.” It never considers what needs to be at
the destination.
Type 2—Environmental Rationalization(∼20%). The model builds secondary justifica-
tions around the wrong answer. “Walking saves fuel and is better for the environment.”
Type 3—Ironic Self-Awareness(∼10%). The model acknowledges that the car needs to be
at the car wash, then still recommends walking. One B_role_only response said: “You can drive
your car through the wash bay when you arrive”—as if the car would get there on its own.
4.4 Latency Analysis
Condition Median Latency vs Baseline
A_bare 4,649ms —
B_role_only 7,550ms +62%
C_role_star 7,851ms +69%
D_role_profile 8,837ms +90%
F_role_star_profile 9,056ms +95%
E_full_stack 8,347ms +80%
Table 2: Median response latencies. E_full_stack is faster than D and F despite having more
context, suggesting that model confidence reduces deliberation time.
5

5 Discussion
5.1 Why STAR Works: The Task Step
The mechanism is in the Task step. Without STAR, the model goes straight from “100 meters” to
the distance heuristic to “walk.” The purpose of the trip—washing the car—is in the input, but the
model has no obligation to process it before reaching a conclusion.
STAR changes the generation sequence. The model must fill in:
Situation: I want to wash my car. The car wash is 100 meters away.
Task: ___
This creates a fork. If the model writes “Task: Get your car to the car wash,” the car becomes
the subject of the goal statement. Drive follows naturally. If the model writes “Task: Get yourself
and your car to the car wash efficiently,” the person re-enters as the subject, and walk becomes
plausible again.
The per-trial data confirms this. All 17 passing trials in C_role_star had Task statements where
the car was the primary subject. All 3 failures framed the task around the person.
Once the model generates “Task: Get your car to the car wash,” every token that follows is
conditioned on that text. The implicit constraint—the car must physically be there—is now explicit
in the context window. STAR does not give the model new information. It makes the model write
down what it already has before moving on.
5.2 Why Profile Injection Alone Falls Short
D_role_profile gave the model everything it needed: Sarah drives a 2022 Honda Civic, it is parked
in the driveway, she is at home. This is enough to answer correctly. The pass rate was 30%.
The problem is not about missing information. The model has the facts. But having facts
in the context window does not mean the model will use them at the right moment. Without a
reasoning structure, the model still takes the shortest path from input to output. “100 meters”
triggers the distance heuristic and the conclusion lands before the car’s location ever gets pulled
into the reasoning chain.
5.3 The Recovery Paradox
C_role_star scored 85% on first pass but only 67% on recovery. A_bare and B_role_only scored
0% on first pass but recovered at 95–100%. F_role_star_profile showed an even more extreme
version: 95% first-pass accuracy but 0% recovery on its single failure.
When conditions A and B fail, they fail with loose, unstructured responses. A challenge prompt
can redirect easily because there is nothing anchoring the wrong answer. When C or F fails, it
fails with a full STAR-structured argument. The model has already walked through Situation,
Task, Action, and Result, and produced a coherent case for walking. Correcting course means
contradicting a structured argument the model just made.
The mechanism is token-level, not psychological. Prior generated text constrains subsequent
generation. This has a practical consequence: if an initial response used structured reasoning,
the follow-up correction needs to target the specific step that went wrong—in this case, the Task
formulation.
6

5.4 The Profile-RAG Decomposition
The addition of condition F resolves a confound present in the original five-condition design. Pre-
viously, the jump from 85% (C) to 100% (E) could not be attributed to any single layer because E
added both profile and RAG simultaneously.
With F at 95%, the decomposition is now clear:
Layer Added Increment Cumulative
STAR (C vs B) +85pp 85%
Profile (F vs C) +10pp 95%
RAG (E vs F) +5pp 100%
Table 3: Per-layer marginal contributions to pass rate.
Profile contributes twice as much as RAG in the final stretch. The mechanism is likely that
profile grounds the STAR framework in concrete physical details (a specific car, a specific location),
which reduces the probability of the Task step being formulated abstractly. RAG adds situational
context (the car needs washing after a road trip) that eliminates the remaining edge case.
5.5 Open Questions
This study measures behavior at the prompt layer. We can see that STAR produces 85% and profile
injection produces 30%. We do not know what happens inside the model to produce this difference.
Which attention heads activate differently when the Task step is present? Does forcing goal
articulation change the activation pattern in a way that is consistent across prompts, or is it specific
to this question? Would the same STAR structure produce the same lift on GPT-4 or Gemini, or
is the effect tied to Claude’s training?
These are mechanistic interpretability questions. What we have is a behavioral result that any
interpretability study could use as a starting point: the same model, the same question, two prompt
conditions, a 55 percentage point gap.
6 Limitations
Single model.Every trial usedclaude-sonnet-4-5-20250929. Whether these patterns hold
across GPT-4o, Gemini, or Mistral is unknown.
Single task.One question, one correct answer. The car wash problem tests implicit physical
constraint reasoning specifically. Whether STAR produces similar gains on temporal constraints,
social context inference, or causal chain reasoning has not been tested.
Sample size.20 runs per condition. The Fisher’s exact test confirms that the C vs D difference
is statistically significant (p= 0.001), but confidence intervals around individual pass rates remain
wide.
Temperature.0.7 was chosen as a reasonable default for introducing variance. We did not
sweep across temperature values.
Distance modification.The original Mastodon question used 50 meters. We used 100, fol-
lowing the benchmark. The longer distance might make the task slightly easier than the original.
DeepSeek version.The model used for pre-experiment prediction was DeepSeek, but we did
not record the exact version or endpoint.
7

Latency overhead.STAR-structured prompts increased median response time by about 69%
over baseline (7,851ms vs 4,649ms).
Challenge prompt bias.The challenge prompt (“How will I get my car washed if I am
walking?”) is leading. A neutral challenge like “Are you sure?” would better isolate self-correction
ability.
F condition timing.Condition F was added after the initial five-condition experiment, ap-
proximately six days later. While the same model version, hyperparameters, and scoring method-
ology were used, we cannot rule out that API-level changes may have introduced minor behavioral
differences.
7 Conclusion
We started this study because we had a result we could not explain. Our system answered a question
correctly when other LLMs did not, and we did not know which part of the system was responsible.
The experiment gave us a clear answer:
Layer Pass Rate Marginal Contribution
Baseline (A, B) 0% —
STAR (C) 85% +85pp
Profile alone (D) 30% +30pp (without STAR)
STAR + Profile (F) 95% +10pp (on top of STAR)
STAR + Profile + RAG (E) 100% +5pp (on top of STAR + Profile)
Table 4: Complete layer progression from baseline to perfect reliability.
STAR reasoning accounts for the overwhelming majority of the improvement. Profile and RAG
are necessary for perfect reliability but insufficient on their own. The mechanism appears to be
goal articulation: when the model is forced to write down what it is trying to accomplish before
it starts reasoning about how, implicit constraints surface as explicit text. Once they are explicit,
autoregressive generation conditions on them.
There is a broader point here. A common pattern in applied AI is to solve reasoning failures by
adding more context—more facts, more profile data, more retrieved documents. Our results suggest
this is the wrong first move. How the model processes information matters more than how much
information it receives. Profile injection with all the right facts scored 30%. Structured reasoning
with no additional facts scored 85%. The difference was statistically significant (p= 0.001).
Or to put it less formally: intelligence is not about how much you hold in your head. It is about
knowing to pick up the keys before you leave the house.
References
McCarthy, J. and Hayes, P.J. (1969). Some Philosophical Problems from the Standpoint of Artificial
Intelligence.Machine Intelligence 4, Edinburgh University Press, pp. 463–502.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D.
(2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models.Proceedings
of ICLR 2023.
8

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D.
(2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.Advances in
Neural Information Processing Systems 35 (NeurIPS 2022).
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023a). ReAct:
Synergizing Reasoning and Acting in Language Models.Proceedings of ICLR 2023.
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T.L., Cao, Y., and Narasimhan, K. (2023b). Tree
of Thoughts: Deliberate Problem Solving with Large Language Models.Advances in Neural
Information Processing Systems 36 (NeurIPS 2023).
Allen, R. (2026). car-wash-evals. GitHub.https://github.com/ryan-allen/car-wash-evals
9