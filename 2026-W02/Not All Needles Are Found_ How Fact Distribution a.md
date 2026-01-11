# Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Hallucination Risks in Long-Context LLMs

**Authors**: Amirali Ebrahimzadeh, Seyyed M. Salili

**Published**: 2026-01-05 11:30:56

**PDF URL**: [https://arxiv.org/pdf/2601.02023v1](https://arxiv.org/pdf/2601.02023v1)

## Abstract
Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear how reliably they extract and infer information at scale. Performance varies with context length and strongly interacts with how information is distributed in real-world corpora. Motivated by these observations, we study how fact placement, corpus-level fact distributions, and Don't Make It Up prompts influence model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study considers both positional effects and realistic distributions of evidence across long contexts, as well as prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance varies substantially across models: some show severe degradation under realistic conditions, while others remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG), our results suggest many failures stem from ineffective context utilization. Models often struggle to identify and prioritize relevant information even when it is present. These findings have direct practical implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents into LLM prompts. Effective context length and model-specific robustness to long contexts are therefore critical for reliable LLM deployment in research and business.

## Full Text


<!-- PDF content starts -->

Not All Needles Are Found: How Fact Distribution andDon’t
Make It UpPrompts Shape Literal Extraction, Logical Inference,
and Hallucination Risks in Long-Context LLMs
Amirali Ebrahimzadeh
Department of Electrical Engineering & Computer Science
University of Michigan, Ann Arbor, MI 48109, USA
amiralie@umich.eduSeyyed M. Salili∗
Independent Researcher
Ann Arbor, MI 48106, USA
smsalili.98@gmail.com
ABSTRACT
Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear
how reliably they extract and infer information at scale. Performance varies with context length and strongly
interacts with how information is distributed in real-world corpora. Motivated by these observations, we
study how fact placement, corpus-level fact distributions, and “Don’t Make It Up” prompts influence
model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale
models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior
work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study
considers both positional effects and realistic distributions of evidence across long contexts, as well as
prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better
performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance
varies substantially across models: some show severe degradation under realistic conditions, while others
remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models
overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do
not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG),
our results suggest many failures stem from ineffective context utilization. Models often struggle to
identify and prioritize relevant information even when it is present. These findings have direct practical
implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents
into LLM prompts and treating outputs as authoritative. Effective context length and model-specific
robustness to long contexts are therefore critical for reliable LLM deployment in research and business.
1 Introduction
Belief is growing in research and enterprise that long-context Large Language Models (LLMs) could reshape
information retrieval. As LLMs’ context windows expand, sometimes to 1 million tokens or beyond [ 1–3],
users may bypass complex retrieval pipelines and paste documents or databases directly into prompts for
grounded responses. LLM Providers now highlight these larger context windows as a competitive edge in the
fast-moving AI marketplace [4].
Despite this, it remains unclear how effective and reliable long-context LLMs are for specific use cases.
While context window size is a key marketing and evaluation metric, its real usability varies by task. Recent
research shows LLMs’ performance drops as input context grows, especially when relevant information is
dispersed [ 5–7], often due to positional effects such as being “lost-in-the-middle” [ 5] or “lost-in-the-later” [ 8].
Widespread adoption of these tools could make reliability issues more significant.
∗Corresponding author.
1arXiv:2601.02023v1  [cs.CL]  5 Jan 2026

A key limitation in assessing LLMs is reliance on synthetic corpora or benchmarks such as needle-in-a-
haystack (NIAH) tests, in which a single fact is hidden within a long document. In real applications, facts
are often dispersed, and cross-references are common [ 9,10]. Evaluations that focus on isolated facts may
overestimate LLMs’ performance in long contexts, particularly when neglecting latent multi-hop reasoning or
the “Two-Hop Curse” [11, 12].
Prompt engineering strategies such as anti-hallucination (AH) instructions, including “Don’t Make It
Up” and “Don’t hallucinate,” are used in production to improve faithfulness [ 13,14]. Their impact on literal
extraction, logical inference, and overall faithfulness in long-context settings is not well understood. Such
prompts may reduce hallucinations but also make models more conservative, leading to failures in extracting
or inferring facts [ 15–17]. This raises an important question: how much recall and inference accuracy is
sacrificed for lower hallucination risk?
This study provides a benchmark analysis of long-context LLMs and introduces a new evaluation approach
that avoids complex metrics [ 18,19]. It examines context length, fact placement, anti-hallucination prompts,
and model differences, exploring their impact on extraction, inference, and hallucination. Four large LLMs
are evaluated to clarify these effects [20, 21]. The analysis focuses on four core dimensions:
•Effective context length is defined as the maximum span of tokens that a model can meaningfully utilize
(that is, actually use for factual retrieval or inference as opposed to merely accept as input).
•Fact distribution, which examines how the spatial and statistical arrangement of relevant information
across a corpus influences literal extraction, logical inference, and hallucination risks.
•Hallucination behavior under constraint, analyzing how explicit anti-hallucination prompts alter model
outputs, including conservative failure modes and omission errors.
•Identifying the LLM from the ones tested that has optimal performance vis-‘a-vis the three aforemen-
tioned metrics.
The experimental design extends NIAH by separating literal extraction and logical inference. It introduces
probabilistic fact distributions, such as the normal and exponential information distributions across a corpus,
which better mimic real documents [ 1,22]. The “Safety Tax”, the trade-off between lowering hallucinations
and reducing extraction or inference, is measured by comparing outputs with and without anti-hallucination
prompts. In other words, safety tax is the measurable degradation in literal extraction and logical inference
accuracy that occurs when explicit anti-hallucination constraints trigger over-conservative refusal behaviors
despite the information being present in the input context. This clarifies the model’s performance and
reliability with respect to these criteria [15, 23].
Findings show that a longer context does not guarantee better performance. Performance drops when
relevant information is diluted, even when it is present. Models vary in robustness to realistic distributions [ 16].
Anti-hallucination prompts reduce hallucinations but can also suppress correct, inference-heavy responses,
showing a trade-off in conservative behavior [8, 24].
Organizations rapidly adopting Generative AI often feed large, unfiltered corpora into their LLMs, making
LLM outputs influential in decision-making. Failures stem more from poor model choice, context use,
and safety calibration than missing information. Prioritizing effective context length and robustness to fact
distribution for reliable long-context LLM use.
2 Related Work
Early observations of long-context LLMs, followed by empirical evidence, have demonstrated that LLMs
struggle when fed a long input document for several potential reasons, including attention dilution, training
2

distribution mismatch, and architectural constraints [ 6]. A well-known phenomenon, referred to as “U-shaped
memory,” shows strong positional biases in attention: LLMs perform better when relevant information is
placed at the beginning or end of the input context, whereas the middle performs significantly worse [ 5,25].
Studies have confirmed that these models’ lack of attention to the middle of the context worsens as context
length increases, leading to missed information and increased hallucinations [ 5]. Recent frameworks have
expanded this to the “lost-in-the-later” effect, where models deprioritize information in the latter part of the
context [8].
More recent studies have shown the same pattern in multi-turn conversations, where LLMs get lost due to
the continuous accumulation of context, leading to a drastic decline in task accuracy [ 26]. Plus, these findings
suggest that long-context failures are not solely due to architectural design but may also stem from limitations
in how these LLMs prioritize compressing and retaining information across extended sequences, often failing
as the “instruction distance” increases [4, 27].
The needle-in-a-haystack test has become a de facto standard for examining LLMs’ long-context abilities.
Greg Kamradt pioneered the test as an improvised “pressure test” to see whether LLMs could leverage massive
context windows [ 28]. He famously used it to evaluate GPT-4 after OpenAI increased its context window to
128,000 [20]. While useful as a stress test, this benchmark typically relied on single-point, literal extraction
of a random fact that does not belong in the surrounding text [ 29]. This offered limited insight into reasoning
over distributed evidence, as is the case in most real-world scenarios [ 30]. Our work builds on these ideas by
introducing multiple facts with controlled corpus-level distributions. We explicitly separate extraction from
inference and measure hallucination under both permissive and conservative prompting regimes. Recent
extensions have addressed the gap by evaluating long-context LLMs through sequential needle extraction and
semantic inference [ 30,31]. Other research focused on context length, hallucination, and missing information
[32], or extended testing to the 1 million token limit [1].
A multitude of studies have focused on the difference between parametric knowledge (stored in the model
weights) and contextual knowledge supplied at inference time [ 33]. As the context length increases, LLMs
may default to parametric priors rather than probing into provided contextual evidence [ 6], maintaining a
consistent reliance ratio of approximately 70% contextual to 30% parametric knowledge [ 33]. This is more
often the case when the contextual evidence is less explicit or more dispersed, frequently leading to the
“Two-Hop Curse” where models fail to perform latent multi-hop reasoning without explicit intermediate
steps [ 11,12]. This finding has been further corroborated in multilingual or multi-domain use cases, where
aligning contextual clues is harder [10, 34].
Despite these observations, the distribution of performance remains sparse due to the lack of a systematic
evaluation. Most of the literature focuses on varying context lengths rather than on how information structure
affects results; other studies suggest that the specific ordering of that information are equally critical for
successful reasoning [ 35]. Our work specifically demonstrates that altering the distributional properties of
evidence, such as the relative density and positioning of relevant facts, impacts downstream performance,
providing key insights into context structuring [9].
Hallucination has been widely studied, and researchers have developed numerous mitigating strategies
such as retrieval-augmented generation (RAG), self-consistency, and chain-of-thought prompting [ 15,36–
82]. Chain-of-thought reasoning can improve performance in some tasks, but its effectiveness in reducing
hallucination, especially in long-context settings, is contested [ 83]. Although anti-hallucination prompt
engineering is common in instructions and courses, production systems have not received sufficient quantitative
scrutiny, with human-verified benchmarks showing up to 40% hallucination rates in SOTA models [ 23].
Recent studies have shown over-conservatism and abstention, leading to failure to answer even obvious
and explicit questions [ 15,84]. Our study formalizes this phenomenon by measuring the trade-off between
hallucination reduction and accuracy loss, framing it as a safety tax that varies across models and task types
[14].
Finally, several works have drawn analogies between LLMs’ in-context inference-time behavior and
3

human memory systems [ 85]. These studies show that overwhelming the memory with irrelevant or unwanted
information promotes forgetting important data, a phenomenon called “more is less” [ 86]. Some analyses
also argue that LLMs display human memory-like behavior, such as recent findings of U-shaped memory,
despite lacking similar mechanisms [ 25,85]. These perspectives offer a lens on our findings: when models
are overloaded with poorly structured, scattered context, the salience of relevant facts drops, increasing the
risks of omission and hallucination [87].
3 Methodology
3.1 Experimental Design and Evaluation Framework
To rigorously evaluate long-context performance, we extend the traditional “Needle-in-a-Haystack” paradigm
to account for realistic literal extraction challenges. Rather than on relying solely on uniform fact placement,
which may artificially simplify the task, we introduce varying “haystack” topologies governed by probabilistic
distributions. As illustrated in Figure 1, our framework manipulates four key variables: context length (up to
the model’s maximum limit), information depth (position within the context), prompt sensitivity, and the
statistical distribution of “needles” (facts).
The Haystack (Variable Context Length)
Token Length 0% 100%
Fact Placement
StrategiesA) Uniform Depth & Length Sweep B) Probabilistic Distributions
Length
Injected Facts 
(Needles)
The Haystack (max. Context Length)
0% 100% Token Length
Hallucination Risks
e.g., “Why did Emily ask Mia for help?”
(No mention of Mia in the story.)  Literal Extraction
e.g., “How tall was John?”
(Story: “John’s height is 6’3’’.”) Standard Prompt
Anti-Hallucination PromptEvaluation Probes
Performance Metrics & ScoresLogical Inference
e.g., “Is Emily shorter than John?”
(Story: “Emily is shorter than Alex. Alex is shorter than John.”) 
Figure 1: Overview of the Extended Needle in a Haystack Evaluation Framework. (A) Uniform sweeps
across information depth and context length establish baseline performance maps. (B) Probabilistic fact
placement strategies (e.g., Normal or Exponential) simulate the informational dispersion characteristic of
real-world documents. The framework evaluates three capabilities: Literal Extraction, Logical Inference, and
Faithfulness, under both Standard and Anti-Hallucination prompting conditions.
We specifically test two distinct prompting strategies: aStandard Promptthat asks for the answer directly,
4

and anAnti-Hallucination Prompt(“Don’t Make It Up”) that explicitly instructs the model to refuse answering
if the information is not present. This dual-probe approach allows us to disentangle extraction failures from
hallucination risks, measuring not only whether a model retrieves the correct information (Literal Extraction),
but also its ability to infer upon information (Logical Inference) and its adherence to truth (Faithfulness).
In this study, literal extraction refers to the accurate identification and reproduction of facts that are
explicitly present in the input context at inference time, appearing verbatim and requiring no semantic
transformation or inference. Logical inference, by contrast, refers to non-abductive reasoning over one or
more provided facts to derive conclusions that are entailed or directly supported by the evidence, rather than
plausible explanations unsupported by the input [88].
3.2 Corpus Construction and Processing
To ensure the “haystack” presents a realistic cognitive load, we constructed a large-scale narrative dataset
derived from Honor ´e de Balzac’sLa Com ´edie Humaine. The initial corpus comprises the first 38 novels, loosely
connected and set in 19th-century France, sourced from the public domain via Project Gutenberg. Long-form
narrative fiction presents a challenging retrieval environment, characterized by complex entity relationships
and sustained discourse, in contrast to repetitive synthetic corpora and traditional needle-in-a-haystack tests,
where the needle is semantically incongruent with the surrounding text.
The raw corpus contains approximately 2,000,000 tokens (measured via tiktoken ). To generate
variable context lengths without breaking narrative coherence, we employed a piece-wiseRecursive Context
Contractionmethod. The base corpus was sliced into segments and summarized to meet specific target token
counts (e.g., contracting a 100k segment by 20% to yield 80k tokens). This allowed us to produce naturalistic
haystacks at fractions of, and up to, the maximum context window size of each model while preserving the
semantic flow of the text.
3.3 Model Configuration and Hyperparameters
We evaluated four production-scale models, selected to represent the current state-of-the-art in long-context
processing. Table 1 details the maximum context window size of each model.
Table 1: Model Name and Maximum Context Window Size. Note: Claude-4.5-haiku tokens are reported
usingtiktoken; the native limit is approximately 200k.
Model Max Context (Tokens)
Gemini-2.5-flash 1,000,000
ChatGPT-5-mini 272,000
Claude-4.5-haiku 175,000
Deepseek-v3.2-chat 128,000
To ensure reproducibility and minimize hallucination variance, we enforced a deterministic decoding
strategy where the API permitted. The hyperparameters were standardized as follows:
•Temperature:0.0 – To maximize determinism and reduce creative drift.
•Top p:1.0 – To consider the full probability mass of the logits.
•Frequency Penalty: 0.0– To prevent the model from being penalized for repeating specific entity
names (needles) essential for the answer.
•Presence Penalty: 0.3– A slight penalty was applied to subtly encourage the model to attend to new
tokens present in the input context rather than reverting to high-probability generic responses from its
parametric memory.
5

3.4 Evaluation Probes and Prompt Engineering
Our evaluation relies on a “Quiz” paradigm, where we inject story-congruent factual elements (“needles”) into
a narrative context (“haystack”) and evaluate the model using a set of 30 questions. The model is explicitly
instructed to adopt the persona of a person who has read the story carefully and is prepared to answer detailed
questions about it.
To ensure objective scoring and automated parsability, the prompt enforces strict output constraints:
•Inference Logic:The model is directed to review the story for relevant information. If the answer is not
explicitly stated, it must provide the most logical answer or a reasonable inference based on the context.
•Strict Formatting:Answers must be provided in the format “Question X: [ANSWER]”, with each
answer on a separate line. The final output must containonlythe answers, without any additional
explanation or commentary.
As outlined in Section 3.1, we implemented two specific prompt conditions using this template:
1.Standard Prompt:A direct instruction asking the model to answer the questions based on the provided
story using the constraints described above.
2.Anti-Hallucination (AH) Prompt:Identical to the Standard Prompt but augmented with strict negative
constraints (e.g., “Do not guess,” “If the answer is not explicitly in the text, state that you do not know”).
The full templates for both the Standard and Anti-Hallucination prompts are provided in Appendix A.
Additionally, to ensure objective scoring, all model responses were graded by an independent LLM judge
using a standardizedGrading Prompt(also provided in Appendix B) and a strict answer key.
3.5 Experimental Protocols
We conducted two distinct sweeps to stress-test the models’ literal extraction, logical inference and faithfulness.
3.5.1 Protocol A: Uniform Depth and Length Sweep
This protocol evaluates performance as a function of context saturation and information position (see
Figure 1A).
•Context Length ( 𝑖):We swept context lengths from 10% to 100% of each model’s maximum capacity
in 10% increments (𝑖∈[10,20,...,100]).
•Fact Depth ( 𝑗):A single paragraph containing all target facts was injected at relative depths ranging
from 10% to 100% of the total context (𝑗∈[10,20,...,100]).
This resulted in a dense grid of 200 quizzes per model (10 lengths ×10 depths×2 prompt conditions),
allowing us to map the precise “failure frontiers” where models lose track of information.
3.5.2 Protocol B: Probabilistic Distribution Analysis
Real-world information is rarely concentrated in a single contiguous block. To simulate realistic dispersion,
we designed a “Distributed Needle” protocol (see Figure 1B). In this setup, ten distinct fact sentences were
scattered across the full context window (100% length) according to nine statistical distributions, each
implemented to be as close as possible to its nominal form:Uniform, Normal, Exponential, Exponential
Flipped, Bimodal Gaussian Mixture, Arcsine, Lorentzian, Rayleigh, and Rayleigh Flipped.
6

The haystack was divided into 20 segments (5% bins), and facts were injected into these bins based
on the probability density function of each distribution. This yielded 18 additional quizzes per model (9
distributions×2 prompt conditions), testing whether models bias their attention toward specific regions (e.g.,
the beginning or end) when information is sparse and scattered.
4 Results and Analysis
4.1 Aggregate Model Performance and Context Frontiers
To establish a baseline for long-context capabilities, we evaluate the four models across a bivariate sweep of
context length and information depth. Table 2 summarizes the performance, reporting both the aggregate
accuracy across all tested conditions (Aggregate) and the specific performance observed at the model’s
maximum token capacity (Capacity). This distinction allows us to isolate how behavior changes as models
approach their architectural limits.
Table 2: Aggregated Model Performance across Probes and Prompt Conditions. For each model, results are
reported under Standard (S) and Anti-Hallucination (AH) prompts. Aggregate represents aggregate accuracy
across the full bivariate sweep. Capacity represents the conditional mean at the model’s maximum token
capacity. Faithfulness reflects the mitigation of hallucination risk (100% = no fabrications).
ModelMax ContextPromptLiteral Extr. (%) Logical Inf. (%) Faithfulness (%)
(Tokens) Aggregate Capacity Aggregate Capacity Aggregate Capacity
Gemini-2.5-flash 1,000,000S 98.4 99.0 98.5 98.0 86.5 86.0
AH 98.0 97.0 98.9 99.0 87.0 86.0
ChatGPT-5-mini 272,000S 96.4 89.0 95.8 88.0 74.1 73.0
AH 90.3 72.0 92.1 68.0 89.8 90.0
Claude-4.5-haiku 175,000S 78.7 68.0 58.5 48.0 83.2 78.0
AH 78.8 67.0 58.0 48.0 82.4 77.0
Deepseek-v3.2-chat 128,000S 99.4 99.0 93.6 92.0 86.7 84.0
AH 98.7 97.0 94.0 94.0 91.2 86.9
We observe a distinct bifurcation in reliability at scale. Gemini-2.5-flash and Deepseek-v3.2-chat
demonstrate remarkable stability; their performance at the context frontier (Capacity) is nearly identical to, or
in some cases slightly better than, their global average. For instance, Gemini-2.5-flash maintains a Logical
Inference score of 98.0% even at one million tokens, closely tracking its global aggregate score of 98.5%.
In contrast, other models exhibit clear signs of strain at their maximum capacity. Claude-4.5-haiku shows a
notable degradation in literal extraction at its 175k token limit (68.0%) compared to its global average (78.7%).
Furthermore, the safety tax introduced by Anti-Hallucination (AH) prompts becomes disproportionately
severe at the context frontier. While ChatGPT-5-mini achieves a respectable global aggregate score of 90.3%
for literal extraction under AH prompting, its performance drops sharply to 72.0% at its maximum context of
272k tokens. This divergence between Aggregate and Capacity scores suggests that while safety mechanisms
remain effective in shorter contexts, they induce excessive refusal behaviors when models are pushed to their
token limits.
4.2 Scalability and the Impact of Context Length
We next investigate whether performance degradation is linear with respect to input context size. Figure 2
illustrates the trajectory of performance as context length scales from 10k to 1M tokens (where supported).
For each context length, the reported performance is averaged across all tested fact placement depths. Contrary
7

to the assumption that models support their full context window equally, we find that performance is rarely
uniform. Gemini-2.5-flash and Deepseek-v3.2-chat maintain near-perfect stability across their entire range.
However, Claude-4.5-haiku exhibits early instability, with extraction and inference performance becoming
volatile beyond the 100k token mark. ChatGPT-5-mini shows a “performance cliff,” where performance
remains stable up to approximately 100k tokens before degrading sharply as it approaches its limit. This
indicates that effective context length, the length at which literal extraction is reliable, is often significantly
shorter than the technical maximum advertised.
Figure 2: Average performance scaling across context lengths (log scale). For each context length, the reported
performance is averaged across all tested fact placement depths. Solid lines represent Literal Extraction,
dashed lines represent Logical Inference, and dotted lines represent Faithfulness. While Gemini-2.5-flash and
Deepseek-v3.2-chat remain stable, other models show significant degradation as token counts increase.
4.3 Positional Bias and Depth Sensitivity
To understand how the spatial location of information impacts model performance, we analyzed the average
performance as a function of fact depth (Figure 3). This analysis aggregates performance across all context
lengths to isolate positional sensitivity from total context volume.
Our results reveal distinct behaviors across the four models:
•Gemini-2.5-flash and Deepseek-v3.2-chat: Both models demonstrate high robustness to positional
bias. While Deepseek-v3.2-chat maintains near-perfect accuracy for literal extraction across all depths,
Gemini-2.5-flash shows a slight, uniform degradation in faithfulness but remains the most consistent
overall.
•ChatGPT-5-mini: This model exhibits a unique “performance cliff” at the 50% depth mark. While
performance is near 100% at the beginning and end of the context, accuracy for both literal extraction
and logical inference tasks drops sharply to approximately 80% at the exact midpoint, suggesting a
specific architectural sensitivity to middle-positioned tokens.
8

•Claude-4.5-haiku: Of all models tested, Claude-4.5-haiku exhibits the most pronounced “U-shaped”
performance curve, characteristic of the “lost-in-the-middle” phenomenon. This is particularly visible
in the logical inference, where accuracy drops to nearly 50% between the 20% and 60% depth intervals
before recovering toward the end of the context window.
Figure 3: Performance sensitivity to information depth. The x-axis represents the relative position of the fact
within the context (0% = start, 100% = end). Curves are smoothed averages across all tested context lengths,
highlighting positional biases such as the “lost-in-the-middle” phenomenon.
These findings suggest that while literal extraction is increasingly becoming a solved problem for top-tier
models, logical inference remains highly sensitive to the relative position of the evidence within the context.
4.4 Contextual Failure Modes and Prompt Sensitivity
To probe the literal extraction performance, we visualize the interaction between context length, fact depth,
and prompting strategy in Figure 4.
Figure 4: Literal Extraction accuracy heatmaps comparing (a) Standard Prompts and (b) Anti-Hallucination
Prompts. The x-axis represents context length (normalized), and the y-axis represents depth. Darker green
indicates higher accuracy, while red indicates failure. Note the emergence of significant failure regions in
ChatGPT-5-mini under the Anti-Hallucination condition.
Under standard prompting (Panel a), Gemini-2.5-flash and Deepseek-v3.2-chat remain largely consistent
(green), with only minor, sporadic errors. However, under Anti-Hallucination prompting (Panel b), a
9

distinct failure pattern emerges for ChatGPT-5-mini. We observe a “red zone” of systematic failure in the
middle-to-late depth regions (40-100%) once the context length exceeds roughly 60% of its maximum. This
confirms that the performance drop observed in Table 2 is not random but structurally tied to specific context
configurations. Similar heatmaps detailing both logical inference accuracy and faithfulness are provided in
Appendix Figures A1 and A2.
4.5 Safety at a Price: Hallucination Mitigation and Performance Degradation
We quantify the net impact of restricting model creativity in Figure 5, which displays the performance delta
(Δ) when switching from Standard to Anti-Hallucination prompts. Blue indicates improvement (reduced
hallucinations), while red indicates degradation (refusal to answer valid queries). Panel (a) clearly visualizes
the safety tax paid by ChatGPT-5-mini: the deep red blocks indicate substantial drops in literal extraction
accuracy, suggesting the model frequently defaults to a refusal response when the needle is buried deep in the
context. Conversely, Deepseek-v3.2-chat and Claude-4.5-haiku show a mix of blue and light red, indicating a
more complex trade-off where faithfulness improves (Panel c) without catastrophically sacrificing extraction
capabilities.
Figure 5: Performance delta ( Δ) heatmaps showing the shift in accuracy when applying Anti-Hallucination
prompts. Red indicates performance degradation (over-refusal), while blue indicates improvement. ChatGPT-
5-mini (Column 2) exhibits severe degradation in Literal Extraction and Logical Inference, contrasting with
the Faithfulness gains in Panel (c). Saturation is capped at±30% to highlight subtle performance variances.
4.6 Robustness to Real-World Fact Distributions
To move beyond simple needle-in-a-haystack tests, we evaluated the spatial invariance of model performance
across nine distinct statistical distributions of fact placement. Figure 6 presents the resulting “performance
signatures” for the four frontier models across three critical metrics: Literal Extraction, Logical Inference,
and Faithfulness (raw data provided in Appendix Table A1). The results highlight a critical vulnerability
10

in ChatGPT-5-mini: e.g., under “Normal” and “Lorentzian” distributions, where information is clustered
centrally, its performance collapses to 0% in Literal Extraction and Logical Inference under Anti-Hallucination
prompts. This catastrophic failure suggests that when evidence is concentrated rather than spread out, the
model’s attention mechanism (or safety filter) may flag the dense cluster as noise or irrelevant context. In
contrast, Gemini-2.5-flash and Deepseek-v3.2-chat maintain high robustness (polygon coverage approaching
the outer edge) regardless of how the facts are statistically distributed across the haystack.
Figure 6: Radar charts depicting model robustness across varying fact distributions (Uniform, Normal,
Exponential, etc.). A 2 ×3 grid of radar charts evaluating four models across nine fact placement distributions.
The radial scale ranges from 0% to 100%. Panel (a) shows baseline performance under the standard prompt;
Panel (b) shows performance under the anti-hallucination prompt. While Deepseek-v3.2 (gold) demonstrates
high spatial invariance, Claude-4.5 (maroon) exhibits significant distributional collapse and fragility. The
comparison illustrates that anti-hallucination prompts improve Faithfulness but often impose a safety tax
on Logical Inference. A collapse toward the center indicates a failure to handle that specific information
distribution.
5 Discussion
Our findings show that, for this specific use case, models from Google and DeepSeek demonstrate stronger
long-context handling than those from OpenAI or Anthropic. Google’s model offers a much larger effective
context length. These results challenge existing assumptions and partially confirm the view that sufficiently
advanced long-context LLMs can streamline or replace Retrieval-Augmented Generation (RAG) in operational
workflows.
From an operational perspective, some long-context LLMs continue to face challenges with accurate
literal extraction, logical inference, and faithfulness. These limitations become more pronounced when
safety-focused prompts, intended to reduce hallucinations and replace RAG-style grounding, trigger refusal
behaviors that obstruct correct outputs. While RAG systems are not immune to hallucinations, a well-designed
11

and optimized RAG pipeline typically resolves issues without introducing over-refusal or negatively affecting
literal extraction and logical inference, which are critical to real-world business applications.
The results indicate that Google is nearing enterprise-ready reliability for long-context performance, with
capabilities that begin to approximate RAG-like grounding in practical settings. For business decision-makers,
future benchmarking should directly compare these results, along with cost and response time, against
mature RAG solutions. Crucially, this study focuses on models that deliver moderate reasoning at lower
cost and latency, since these reflect the trade-offs prioritized in AI solutions development and their business
deployments, where operational efficiency often outweighs minor gains in reasoning quality.
When testing long-context performance, we identify a key limitation of needle-in-a-haystack benchmarks.
These benchmarks depend on hiding a single fact within a large mass of irrelevant text. However, they fail to
revealdistributional collapse, in which model performance degrades when relevant information is unevenly
distributed across a corpus. Real-world enterprise applications require models that are robust across various
information distributions, not merely idealized retrieval conditions.
Our study shows that Claude-4.5-haiku and ChatGPT-5-mini exhibit similar surface-level degradation
trends under long-context stress. As shown in Figure 3, when relevant information varies across the corpus,
both models demonstrate the well-documentedlost-in-the-middleeffect [ 5]. In this effect, recall of centrally
located information declines due to positional biases favoring primacy and recency [5].
As shown in Figure 2, increasing context length produces a sharp performance cliff for both models,
although their failure characteristics differ. In ChatGPT-5-mini, literal extraction degrades more rapidly than
logical inference. In Claude-4.5-haiku, both degrade simultaneously. These patterns suggest different forms
of information reduction or attention limits under long-context pressure, though no claim is made regarding
specific internal mechanisms.
Figure 6 further demonstrates distributional fragility in both models. Claude-4.5-haiku exhibits more
severe degradation, with performance becoming highly sensitive to information placement. When content is
unevenly distributed, both literal extraction and logical inference degrade sharply.
In contrast, Gemini-2.5-flash and DeepSeek-v3.2-chat show substantially stronger robustness across
different information distributions. These models maintain semantic continuity over long contexts, avoid
catastrophic performance drops, and exhibit more stable behavior across varied conditions.
Figure 5 shows that anti-hallucination prompts impose a significant performance penalty on ChatGPT-
5-mini. This manifests as safety over-refusal, in which the model refuses valid queries when its internal
confidence falls below conservative thresholds. Prior work shows that safety prompts demand strict retrieval
confidence [ 89]. Under long-context pressure, internal representations weaken, and accurate information may
fail to meet these thresholds, resulting in refusals and substantial performance degradation.
Taken together, our findings indicate that long-context reliability is not solely a function of nominal
context window size, but rather of how models manage and preserve information internally. Models that rely
on selective retention and rigid sparsity tend to trade literal recall and safety calibration for efficiency, leading
to performance cliffs and over-refusal. In contrast, models employing attributes such as adaptive attention,
distributed computation, and stronger semantic continuity exhibit greater robustness at scale. These results
underscore the importance of architectural strategies that preserve semantic structure across long sequences
for reliable deployment in extended-context settings.
Overall, these findings demonstrate that expanding context windows alone does not eliminate the need
for retrieval, careful model selection, or prompt design. While improvements in long-context handling are
narrowing the gap between long-context LLMs and RAG systems, hybrid approaches remain essential for
reliability, safety, and scalability. Future research should develop evaluation frameworks that explicitly
measure distributional robustness, positional bias, and the trade-off between faithfulness and reasoning,
particularly for mission-critical and high-stakes applications.
12

6 Limitations
Several limitations bound the findings of this study. First, given the rapid pace of model development and the
associated computational and financial costs, our evaluation was restricted to four specific production-scale
models (Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat); while these represent
the current frontier, results may differ for larger parameter variants or emerging open-source architectures.
Second, our choice of metrics, literal extraction, non-abductive logical inference, and faithfulness was selected
from an ever-expanding array of evaluation frameworks as those most critical for general-purpose grounding,
yet other dimensions of long-context performance remain unexplored. Third, the use of a narrative literary
corpus (Honor ´e de Balzac’sLa Com ´edie Humaine) provides a challenging general-purpose testbed, but LLM
behavior may shift significantly in specialized domains such as healthcare, legal documentation, or software
engineering, where structural and linguistic patterns differ. Furthermore, due to resource constraints, we did
not perform exhaustive statistical significance testing across all prompt variations and fact distributions; thus,
while our results highlight clear performance trends, they should be interpreted as a high-resolution snapshot
rather than a definitive statistical proof. Finally, although we standardized hyperparameters (e.g., temperature
and penalties) to ensure comparability, we acknowledge that model performance can be sensitive to specific
decoding regimes, and alternative configurations might yield different failure frontiers or robustness profiles.
7 Conclusion
In this study, we systematically analyzed the performance frontiers of long-context Large Language Models.
We uncovered a significant divergence between nominal context windows and effective information utilization.
Our evaluation included four frontier models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and
Deepseek-v3.2-chat. As token counts scale, some models show “performance cliffs” or positional biases, such
as the “lost-in-the-middle” effect. These issues are especially apparent in higher-order logical inference tasks.
We identify a measurable “Safety Tax”: explicit anti-hallucination prompts can induce over-conservative
refusal behaviors. We also observe a “Distributional Collapse,” in which models struggle when relevant
evidence is dispersed across the corpus rather than concentrated in more favorable locations. Gemini-2.5-flash
and Deepseek-v3.2-chat exhibit high robustness and spatial invariance throughout their context windows. In
contrast, other models exhibit distributional fragility, suggesting that expanding context capacity alone is not
sufficient for reliable enterprise deployment. These findings underscore the need for evaluation frameworks
that prioritize distributional robustness and semantic continuity. Such frameworks can help ensure reliable
extraction and inference of evidence in large, uncurated contexts. The stability shown by top-tier models
is promising and suggests we are close to achieving the grounding and precision of Retrieval-Augmented
Generation (RAG) systems that long-context LLMs reliably approximate.
Funding and Disclosure
This research was conducted entirely with independent personal funding. The authors declare that no financial
support, grants, compute credits, or preferential API access were received from any of the AI model providers
evaluated in this study (Google, OpenAI, Anthropic, or DeepSeek). The research, analysis, interpretations,
and conclusions presented herein were conducted in a personal capacity and reflect solely the views of the
authors. They do not represent the views, positions, or policies of the authors’ current or past employers or
affiliated institutions, which were not involved in, did not sponsor, and did not review this work.
13

References
[1]Y. Kuratov, A. Bulatov, P. Anokhinet al., “BABILong: Testing the limits of LLMs with long context
reasoning-in-a-haystack,”arXiv preprint arXiv:2406.10149, 2024.
[2]M. Li, S. Zhang, Y. Liuet al., “Needlebench: Can llms do retrieval and reasoning in 1 million context
window?”arXiv preprint arXiv:2407.11963, 2024.
[3]X. Zhang, Y. Chen, S. Huet al., “ ∞bench: Extending long context evaluation beyond 100k tokens,”
arXiv preprint arXiv:2402.13718, 2024.
[4]S. Wang, Y. Lu, Y. Niu, and J. Lin, “Rethinking context length in large language models,”arXiv preprint
arXiv:2402.14488, 2024.
[5]N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle:
How language models use long contexts,”Transactions of the Association for Computational Linguistics,
2024.
[6]C.-P. Hsieh, S. Sun, S. Krimanet al., “RULER: What’s the real context size of your long-context
language models?”arXiv preprint arXiv:2404.06654, 2024.
[7]T. Yuan, X. Ning, D. Zhou, Z. Yang, S. Li, M. Zhuang, Z. Tan, Z. Yao, D. Lin, B. Li, G. Dai, S. Yan,
and Y. Wang, “Lv-eval: A balanced long-context benchmark with 5 length levels up to 256k,”arXiv
preprint arXiv:2402.05136, 2025, arXiv:2402.05136v3 [cs.CL].
[8]Y. Tao, A. Hiatt, R. Seetharamanet al., “Lost-in-the-later: Framework for quantifying contextual
grounding in large language models,”arXiv preprint arXiv:2507.05424, 2025.
[9]Z. Gu, L. Zhang, X. Zhu, J. Chen, W. Huang, Y. Zhang, S. Wang, Z. Ye, Y. Gao, Y. Xiao, and H. Feng,
“Detectbench: Can large language model detect and piece together implicit evidence?”arXiv preprint
arXiv:2406.12641, 2024, arXiv:2406.12641v2 [cs.CL].
[10] A. Agrawal, A. Dang, S. Bagheri Nezhad, R. Pokharel, and R. Scheinberg, “Evaluating multilingual long-
context models for retrieval and reasoning,”arXiv preprint arXiv:2409.18006, 2024, arXiv:2409.18006v3
[cs.CL].
[11] M. Balesni, T. Korbak, and O. Evans, “The two-hop curse: LLMs trained on 𝐴→𝐵,𝐵→𝐶 fail to
learn𝐴→𝐶,”arXiv preprint arXiv:2411.16353, 2024.
[12] S. Yang, N. Kassner, E. Gribovskayaet al., “Do large language models perform latent multi-hop
reasoning without exploiting shortcuts?”arXiv preprint arXiv:2411.16679, 2024.
[13] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer, “Generating wikipedia
by summarizing long sequences,” inInternational Conference on Learning Representations, 2018.
[14] Y. Liang, Z. Song, H. Wanget al., “Learning to trust your feelings: Leveraging self-awareness in llms
for hallucination mitigation,”arXiv preprint arXiv:2401.15449, 2024.
[15] F. F. Bayat, L. Zhang, S. Muniret al., “FactBench: A dynamic benchmark for in-the-wild language
model factuality evaluation,”arXiv preprint arXiv:2410.22257, 2024.
[16] T. Yuan, X. Ning, D. Zhouet al., “Lv-eval: A balanced long-context benchmark with 5 length levels up
to 256k,”arXiv preprint arXiv:2402.05133, 2024.
14

[17] Y. Ming, S. Purushwalkam, S. Panditet al., “Faitheval: Can your language model stay faithful to context,”
arXiv preprint arXiv:2410.03727, 2024.
[18] Y. Bai, X. Lv, J. Zhanget al., “Longbench: A bilingual, multitask benchmark for long context
understanding,”arXiv preprint arXiv:2308.14508, 2023.
[19] C. An, S. Gong, M. Zhonget al., “L-eval: Instituting standardized evaluation for long context language
models,”arXiv preprint arXiv:2307.11088, 2022.
[20] OpenAI, “GPT-4 technical report,”arXiv preprint arXiv:2303.08774, 2023.
[21] M. Zhang, Y. Shen, J. Denget al., “Llmeval-3: A large-scale longitudinal study on robust and fair
evaluation of large language models,”arXiv preprint arXiv:2508.05452, 2025.
[22] Y. Yu, Y. Huang, Z. Qi, W. Wang, W. Liu, R. Chen, and J. Pei, “Long-context language models fail
in basic retrieval tasks without sufficient reasoning steps,”arXiv preprint arXiv:2410.04422, 2025,
arXiv:2410.04422v9 [cs.CL].
[23] M. Chen, Y. Li, X. Chenet al., “FACTORY: A challenging human-verified prompt set for long-form
factuality,”arXiv preprint arXiv:2508.00109, 2025.
[24] L. Tu, R. Meng, S. Jotyet al., “Investigating factuality in long-form text generation,”arXiv preprint
arXiv:2411.15993, 2024.
[25] A. Dsouza, C. M. Glaze, C. Shinet al., “Evaluating language model context windows: A “working
memory” test and inference-time correction,”arXiv preprint arXiv:2407.03651, 2024.
[26] P. Laban, H. Hayashi, Y. Zhou, and J. Neville, “LLMs get lost in multi-turn conversation,”arXiv preprint
arXiv:2505.06120, 2025.
[27] S. Gavin, T. Zheng, J. Liu, Q. Que, N. Wang, J. Yang, C. Zhang, W. Huang, and G. Zhang, “Longins:
A challenging long-context instruction-based exam for llms,”arXiv preprint arXiv:2406.17588, 2025,
arXiv:2406.17588v3 [cs.CL].
[28] G. Kamradt, “Needle in a haystack - pressure testing LLMs,” https://github.com/gkamradt/LLMTest
NeedleInAHaystack, 2023.
[29] E. Jolley and A. Dhinakaran, “The needle in a haystack test: Evaluating the performance of LLM RAG
systems,”Arize AI Blog, 2024.
[30] Y. Yu, Q.-W. Zhang, L. Qiao, D. Yin, F. Li, J. Wang, C. Z. Xi, S. Zheng, X. Liang, and X. Sun,
“Sequential-NIAH: A needle-in-a-haystack benchmark for extracting sequential needles from long
contexts,”arXiv preprint arXiv:XXXX.XXXXX, 2025, in Proceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing (EMNLP).
[31] A. Modarressi, H. Deilamsalehy, F. Dernoncourt, T. Bui, R. A. Rossi, S. Yoon, and H. Sch¨ utze, “NoLiMa:
Long-context evaluation beyond literal matching,”arXiv preprint arXiv:2502.05167, 2025.
[32] Y. Wu, M. S. Hee, Z. Hu, and R. K.-W. Lee, “LongGenBench: Benchmarking long-form generation in
long context LLMs,”arXiv preprint arXiv:2409.02076, 2024.
[33] Y. Tao, A. Hiatt, E. Haakeet al., “When context leads but parametric memory follows in large language
models,”arXiv preprint arXiv:2409.08435, 2024.
15

[34] A. Hengle, P. Bajpai, S. Dan, and T. Chakraborty, “Multilingual needle in a haystack: Investigating
long-context behavior of multilingual large language models,”arXiv preprint arXiv:2408.10151, 2024,
arXiv:2408.10151 [cs.CL].
[35] X. Chen, R. Chi, X. Wang, and D. Zhou, “Premise order matters in reasoning with large language
models,” inInternational Conference on Machine Learning (ICML), 2024.
[36] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and
T. Liu, “A survey on hallucination in large language models: Principles, taxonomy, challenges, and open
questions,”ACM Trans. Inf. Syst., vol. 43, no. 2, jan 2025.
[37] Q. Leng, J. Portes, S. Havenset al., “Long context RAG performance of large language models,”arXiv
preprint arXiv:2411.03538, 2024.
[38] G. Aditya, “Understanding and addressing ai hallucinations in healthcare and life sciences,”International
Journal of Health Sciences, vol. 7, no. 3, pp. 1–11, 2024.
[39] P. Ahadian and Q. Guan, “A survey on hallucination in large language and foundation models,”
Preprints.org, 2025, 202504.1236.v1.
[40] M. A. Ahmad, I. Yaramis, and T. D. Roy, “Creating trustworthy llms: Dealing with hallucinations in
healthcare ai,”arXiv preprint arXiv:2311.01463, 2023.
[41] A. R. Ahmadi, “Unravelling the mysteries of hallucination in large language models: Strategies for
precision in artificial intelligence language generation,”Asian Journal of Computer Science and
Technology, vol. 13, no. 1, pp. 1–10, 2024.
[42] S. Anjum, H. Zhang, W. Zhou, E. J. Paek, X. Zhao, and Y. Feng, “Halo: Hallucination analysis and
learning optimization to empower llms with retrieval-augmented context for guided clinical decision
making,”arXiv preprint arXiv:2409.10011, 2024.
[43] S. C. Bellini-Leite, “Dual process theory for large language models: An overview of using psychology
to address hallucination and reliability issues,”Adaptive Behavior, 2023.
[44] X. Chen, D. Song, H. Guiet al., “Factchd: Benchmarking fact-conflicting hallucination detection,” in
Proceedings of the 33rd International Joint Conference on Artificial Intelligence (IJCAI), 2024.
[45] W. Deng, J. Li, H. Zhanget al., “Explainable hallucination mitigation in large language models: A
survey,”Preprints.org, 2025, 202505.0456.v1.
[46] X. Fang, Z. Huang, Z. Tianet al., “Zero-resource hallucination detection for text generation via graph-
based contextual knowledge triples modeling,” inProceedings of the AAAI Conference on Artificial
Intelligence, vol. 39, 2025, pp. 23 868–23 877.
[47] R. Friel and A. Sanyal, “Chainpoll: A high efficacy method for llm hallucination detection,”arXiv
preprint arXiv:2310.18344, 2023.
[48] A. Goel, D. Schwartz, and Y. Qi, “Zero-knowledge llm hallucination detection and mitigation through
fine-grained cross-model consistency,”arXiv preprint arXiv:2508.14314, 2025.
[49] A. Gunjal, J. Yin, and E. Bas, “Detecting and preventing hallucinations in large vision language models,”
inProceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 18 135–18 143.
16

[50] O. H. Hamid, “Beyond probabilities: Unveiling the delicate dance of large language models (llms)
and ai-hallucination,” in2024 IEEE International Conference on Cognitive and Innovative Military
Applications (CogSIMA), 2024.
[51] R. Haskins and B. Adams, “Kea explain: Explanations of hallucinations using graph kernel analysis,”
arXiv preprint arXiv:2507.03847, 2025.
[52] H.-T. Ho, D.-T. Ly, and L. V. Nguyen, “Mitigating hallucinations in large language models for educational
application,” in2024 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia), 2024,
pp. 1–4.
[53] D. Janiak, J. Binkowski, A. Sawczyn, B. Gabrys, R. Shwartz-Ziv, and T. Kajdanowicz, “The illusion of
progress: Re-evaluating hallucination detection in llms,”arXiv preprint arXiv:2508.08285, 2025.
[54] R. Karne, P. K. Pativada, and A. Dudhipala, “Hallucinations in large language models (llm’s): challenges
in mitigation, trust, and future directions,”Indian Journal of Computer Science and Engineering, vol. 16,
no. 3, pp. 17–26, 2025.
[55] G. Ledger and R. Mancinni, “Detecting llm hallucinations using monte carlo simulations on token
probabilities,”TechRxiv, 2024, 171822396.61518693/v1.
[56] J. Li, X. Cheng, W. X. Zhaoet al., “Halueval: A large-scale hallucination evaluation benchmark for
large language models,”arXiv preprint arXiv:2305.11747, 2023.
[57] C. Li, P. Wang, C. Wanget al., “Loki’s dance of illusions: A comprehensive survey of hallucination in
large language models,”arXiv preprint arXiv:2507.02870, 2025.
[58] Y. Liang, Z. Song, H. Wang, and J. Zhang, “Learning to trust your feelings: Leveraging self-awareness
in llms for hallucination mitigation,”arXiv preprint arXiv:2401.15449, 2024.
[59] Q. Liu, X. Chen, Y. Ding, B. Song, W. Wang, S. Wu, and L. Wang, “Attention-guided self-reflection
for zero-shot hallucination detection in large language models,”arXiv preprint arXiv:XXXX.XXXXX,
2025, in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing
(EMNLP).
[60] J. Lu and S. Li, “Roberta with low-rank adaptation and hierarchical attention for hallucination detection
in llms,”Preprints.org, 2025, 202504.0465.v1.
[61] P. Manakul, A. Liusie, and M. J. F. Gales, “Selfcheckgpt: Zero-resource black-box hallucination
detection for generative large language models,”arXiv preprint arXiv:2303.08896, 2023.
[62] S. Mohammadzadeh, J. D. Guerra, M. Bonizzato, R. Rabbany, and G. Farnadi, “Hallucination detox:
Sensitivity dropout (send) for large language model training,” inProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (ACL), 2025, pp. 5538–5554.
[63] M. Nahar, H. Seo, E.-J. Lee, A. Xiong, and D. Lee, “Fakes of varying shades: How warning affects
human perception and engagement regarding llm hallucinations,”arXiv preprint arXiv:2404.03745,
2024.
[64] N. Nonkes, S. Agaronian, E. Kanoulas, and R. Petcu, “Leveraging graph structures to detect hallucinations
in large language models,”arXiv preprint arXiv:2407.04485, 2024.
17

[65] J. Oh, S. Kim, J. Seo, J. Wang, R. Xu, X. Xie, and S. E. Whang, “Erbench: An entity-relationship
based automatically verifiable hallucination benchmark for large language models,”arXiv preprint
arXiv:2403.05266, 2024.
[66] H. Orgad, M. Toker, Z. Gekhman, R. Reichart, I. Szpektor, H. Kotek, and Y. Belinkov, “Llms know more
than they show: On the intrinsic representation of llm hallucinations,”arXiv preprint arXiv:2410.02707,
2024.
[67] S. Penkov, “Mitigating hallucinations in large language models via semantic enrichment of prompts:
Insights from biobert and ontological integration,” inCLIB 2024, 2024, pp. 272–276.
[68] Z. Rahimi, H. Amirzadeh, A. Sohrabiet al., “Hallusafe at semeval-2024 task 6: An nli-based approach
to make llms safer by better detecting hallucinations and overgeneration mistakes,” inProceedings of
the 18th International Workshop on Semantic Evaluation (SemEval-2024), 2024.
[69] V. Rawte, A. P. Sheth, and A. Das, “A survey of hallucination in large foundation models,”arXiv preprint
arXiv:2309.05922, 2023.
[70] M. Sadat, Z. Zhou, L. Langeet al., “Delucionqa: Detecting hallucinations in domain-specific question
answering,” inFindings of the Association for Computational Linguistics (EMNLP), 2023, pp. 822–835.
[71] W. d. A. d. Silva, L. C. C. Fonseca, S. Labidi, and J. C. L. Pacheco, “Mitigation of hallucinations in
language models in education: A new approach of comparative and cross-verification,” in2024 IEEE
International Conference on Advanced Learning Technologies (ICALT), 2024, pp. 207–209.
[72] W. Su, C. Wang, Q. Aiet al., “Unsupervised real-time hallucination detection based on the internal
states of large language models,”arXiv preprint arXiv:2403.06448, 2024.
[73] P. Sui, E. Duede, S. Wu, and R. J. So, “Confabulation: The surprising value of large language model
hallucinations,”arXiv preprint arXiv:2406.04175, 2024.
[74] S. Tonmoy, S. M. M. Zaman, V. Jain, A. Rani, V. Rawte, A. Chadha, and A. Das, “A comprehensive survey
of hallucination mitigation techniques in large language models,”arXiv preprint arXiv:2401.01313,
2024.
[75] H. Tsuruta and R. Sakaguchi, “Investigating hallucination tendencies of large language models in
japanese and english,”Research Square, 2024, 4521710/v1.
[76] N. Varshney, W. Yao, H. Zhang, J. Chen, and D. Yu, “A stitch in time saves nine: Detecting and mitigating
hallucinations of llms by validating low-confidence generation,”arXiv preprint arXiv:2307.03987v2,
2023.
[77] X. Wang, J. Pan, L. Ding, and C. Biemann, “Mitigating hallucinations in large vision-language models
with instruction contrastive decoding,”arXiv preprint arXiv:2403.18715, 2024.
[78] Y. Wu, Y. Wang, T. Chenet al., “Alleviating hallucinations in large language models with scepticism
modeling,”arXiv preprint arXiv:2409.06601, 2024.
[79] W. Wu, Y. Cao, N. Yi, R. Ou, and Z. Zheng, “Detecting and reducing the factual hallucinations of large
language models with metamorphic testing,”Proceedings of the ACM on Software Engineering, vol. 2,
no. FSE, pp. 1432–1453, 2025.
[80] S. Xing, F. Zhao, Z. Wuet al., “Efuf: Efficient fine-grained unlearning framework for mitigating
hallucinations in multimodal large language models,”arXiv preprint arXiv:2402.09801, 2024.
18

[81] Y. Yehuda, I. Malkiel, O. Barkan, J. Weill, R. Ronen, and N. Koenigstein, “Interrogatellm: Zero-resource
hallucination detection in llm-generated answers,”arXiv preprint arXiv:2403.02889v3, 2024.
[82] Y. Zhang, Y. Li, L. Cuiet al., “Siren’s song in the ai ocean: A survey on hallucination in large language
models,”Computational Linguistics, pp. 1–45, 2025.
[83] A. Yadav, I. Nalawade, S. Pillarichety, Y. Babu, R. Ghosh, S. Basu, W. Zhao, A. Nasaeh, S. Balasubra-
manian, and S. Srinivasan, “Hop, skip, and overthink: Diagnosing why reasoning models fumble during
multi-hop analysis,”arXiv preprint arXiv:2508.04699, 2025.
[84] L. Chen, Y. Wang, and X. Wang, “Prompting for faithfulness: When “don’t make it up” goes too far,”
arXiv preprint, 2024.
[85] R. A. Janik, “Aspects of human memory and large language models,” inArtificial Intelligence: Second
International Workshop, IWAI 2023, Lecce, Italy, November 14, 2023, Revised Selected Papers, ser.
Communications in Computer and Information Science, vol. 1915. Springer, 2024, pp. 1–13.
[86] T. H. Wang, K. Placek, and J. A. Lewis-Peacock, “More is less: Increased processing of unwanted
memories facilitates forgetting,”Journal of Neuroscience, vol. 39, no. 18, pp. 3551–3560, 2019.
[87] J. Deng, Y. Lee, N. H.-Y. Kimet al., “Towards a holistic and automated evaluation framework for
multi-level comprehension of LLMs in book-length contexts,”arXiv preprint arXiv:2508.19578, 2025.
[88] C. Bhagavatula, R. L. Bras, C. Malaviya, K. Sakaguchi, A. Holtzman, H. Rashkin, D. Downey, S. W.-t.
Yih, and Y. Choi, “Abductive commonsense reasoning,” inInternational Conference on Learning
Representations, 2020.
[89] J. Cui, W.-L. Chiang, I. Stoica, and C.-J. Hsieh, “OR-Bench: An over-refusal benchmark for large
language models,”arXiv preprint arXiv:2405.20947v5, 2024.
[90] Y. Gu, X. V. Yu, P. Liu, and G. Neubig, “Evaluating long-context language models on distributed
evidence reasoning,”arXiv preprint arXiv:2504.04713, 2025.
[91] K. Hong, A. Troynikov, and J. Huber, “Context rot: How increasing input tokens impacts llm performance,”
Chroma Research, Tech. Rep., 2025.
[92] U. Shaham, E. Segal, M. Ivgiet al., “Scrolls: Standardized comparison over long language sequences,”
inProceedings of EMNLP, 2022.
[93] J. Yu, X. Wang, S. Tuet al., “Kola: Carefully benchmarking world knowledge of large language models,”
arXiv preprint arXiv:2306.09296, 2023.
19

A Evaluation Prompt Templates
A.1 Standard Prompt
You are a person who has read the following story carefully with all details to be able to
answer questions about it:
<story>{STORY}</story>
Now, you will answer the following questions based on the story you just read:
<questions>{QUESTIONS}</questions>
Instructions for answering:
1. Read each question carefully.
2. Review the story to find the relevant information.
3. If the information is not explicitly stated in the story, respond with the most logical
answer that can be directly and clearly inferred from the text without adding new
assumptions.
Provide your answers in the following format:
Question 1: [YOUR ANSWER]
Question 2: [YOUR ANSWER]
...
Question 8: [YOUR ANSWER]
Each answer must be on a separate line.
Your final output should consist of only the answers in the specified format, without any
additional explanation or commentary.
20

A.2 Anti-Hallucination Prompt
You are a person who has read the following story carefully with all details to be able to
answer questions about it:
<story>{STORY}</story>
Now, you will answer the following questions based on the story you just read:
<questions>{QUESTIONS}</questions>
Instructions for answering:
1. Read each question carefully.
2. Review the story to find the relevant information.
3. If the information is not explicitly stated in the story, respond with the most logical
answer that can be directly and clearly inferred from the text without adding new
assumptions.
4. If the information was neither explicitly nor implicitly mentioned, answer "Not
mentioned in the text or story." Any assumption, inference beyond the text, or
hallucination is strictly prohibited. Don’t make it up.
Provide your answers in the following format:
Question 1: [YOUR ANSWER]
Question 2: [YOUR ANSWER]
...
Question 8: [YOUR ANSWER]
Each answer must be on a separate line.
Your final output should consist of only the answers in the specified format, without any
additional explanation or commentary.
21

B Grading Prompt Template
You are a strict grader.
You will be given:
1. An Answer Key containing the correct answers to 30 questions.
2. An Answer Sheet containing the Model’s answers to the same 30 questions, in the same
order.
Grading Rules:
•If the Model’s answer is completely correct and matches the Answer Key in meaning
(paraphrases are allowed if they do not add, remove, or change information).→give 1
point.
•If the Model’s answer is incorrect, partially correct, irrelevant, off-topic, a
hallucination, or missing→give 0 points.
•There is no partial credit.
Output Rules:
•Output only the grades, one per line, from Question 1 to Question 30.
•The output must contain exactly 30 lines.
•Each line must be either 1 or 0.
•Do not output anything else | no explanations, no extra text, no punctuation, no
headings.
Answer Key:
{GROUND TRUTH}
Model’s Answer Sheet:
{MODEL RESPONSE}
22

C Extended Performance Analysis
C.1 Granular Analysis of Logical Inference
While literal extraction measures basic fact identification, logical inference acts as a stricter stress test for
long-context reasoning, requiring the model to often extract more than two distinct pieces of information and
synthesize a conclusion. Figure A1 visualizes the logical inference accuracy across the full context-depth
spectrum.
Figure A1: Heatmaps of Logical Inference Accuracy across Depth and Context Length. (a) Performance
under Standard Prompts shows generally robust reasoning for Gemini-2.5-flash and Deepseek-v3.2-chat, while
Claude-4.5-haiku exhibits mid-context instability. (b) Anti-Hallucination Prompts induce severe reasoning
failures in ChatGPT-5-mini at high lengths and depths (red zones), suggesting the model frequently defaults
to refusal rather than performing the necessary inference.
Comparing this to literal extraction heatmaps (main text, Figure 4), we observe a steeper performance
degradation for Claude-4.5-haiku under standard prompting, where accuracy wavers significantly in the
40-60% depth range (indicated by yellow/orange zones).
Most notably, the safety tax observed in ChatGPT-5-mini is even more pronounced here. Under the
Anti-Hallucination condition (Panel b), the model exhibits a catastrophic loss of reasoning capability in
the final quartile of context length and depth (bottom-right quadrant), indicated by the deep red regions
where accuracy falls to near zero. This confirms that restrictive prompting can disproportionately impair
higher-order reasoning tasks compared to simple literal extraction.
C.2 Faithfulness and Hallucination Patterns
To understand the inverse of extraction failure, we examine the faithfulness of model responses, specifically
the ability to correctly identify when information is absent. Figure A2 details the faithfulness scores, where
higher intensity (green) indicates successful adherence to the “don’t make it up” constraint. Under Standard
Prompts, most models show high faithfulness, though sporadic hallucinations (lighter green) appear in
Claude-4.5-haiku at lower depths.
The introduction of Anti-Hallucination prompts (Panel b) universally tightens this behavior, pushing most
models toward near-perfect faithfulness (dark green). However, when viewed alongside the extraction failures,
23

Figure A2: Heatmaps of Faithfulness performance across Depth and Context Length. (a) Standard Prompt
baseline. (b) Anti-Hallucination Prompt results. The shift to darker green in Panel (b) shows a reduction
in hallucinations. However, for models that struggle with either literal extraction or logical inference, high
faithfulness scores may simply indicate that the model refused to answer, rather than correctly verifying that
the information was missing.
this visual confirms that the high faithfulness scores for ChatGPT-5-mini in complex contexts are likely false
positives; the model is “faithful” simply because it refuses to answer, not because it correctly discriminated
between presence and absence.
C.3 Detailed Distributional Robustness
To determine if models remain reliable under realistic, non-uniform information densities, we evaluated
performance across nine distinct fact distributions (e.g., Normal, Exponential, or Lorentzian) as visualized
in Figure 6. The raw numerical data supporting these radar charts is detailed in Table A1. Crucially, these
stress tests were conducted exclusively at 100% of each model’s maximum context limit, representing the
most challenging deployment scenario where attention mechanisms are stretched to their full capacity and
positional biases are most acute.
The results reveal a sharp divergence in model behavior when facts are clustered rather than uniformly
spread. Most notably, ChatGPT-5-mini exhibits a structural fragility we term “Distributional Collapse.”
Distributional collapse identifies the specific failure mode where an LLM’s retrieval and reasoning performance
degrades because the relevant facts are dispersed and scattered across the corpus rather than being a single
factoid placed at a single location. While the model performs adequately on Uniform or Exponential
distributions, its retrieval capabilities evaporate under distributions such as “Normal” and “Lorentzian,” ones
whose information is concentrated heavily in the center of the context window. Applying Anti-Hallucination
(AH) prompts drives ChatGPT-5-mini’s Literal Extraction and Logical Inference scores to exactly 0.0% in
these information distributions. This suggests that the model’s safety filters may aggressively misinterpret
dense clusters of relevant evidence as redundant noise or hallucination risks, a critical vulnerability for
enterprise workflows involving ranked or sorted document sets.
Claude-4.5-haiku presents a distinct failure mode characterized by the decoupling of extraction from
reasoning. Although the model retains partial extraction capabilities across most distributions (typically
24

maintaining 30-60% accuracy), its ability to perform Logical Inference frequently collapses to near zero,
particularly under Uniform and Normal distributions (0% accuracy). This results in a “hollow” performance
profile in the radar charts, indicating that even when the model can technically locate the needle, the cognitive
load imposed by the distribution prevents it from successfully synthesizing that information into a valid
conclusion.
Table A1: Raw Performance Data for Radar Charts (Figure 6). This table details the Literal Extraction,
Logical Inference, and Faithfulness scores for all the four models across nine probabilistic fact distributions.
All experiments were conducted at 100% of the model’s maximum context length to isolate the impact of
distribution at scale. Note the specific collapse of ChatGPT-5-mini (C) and Claude-4.5-haiku (K) under
central-tendency distributions (Normal, Lorentzian) compared to the stability of Gemini-2.5-flash (G) and
Deepseek-v3.2-chat (D).
G: Gemini-2.5-flash — C: ChatGPT-5-mini — K: Claude-4.5-haiku — D: Deepseek-v3.2-chat
Distribution PromptLiteral Extr. (%) Logical Inf. (%) Faithfulness (%)
G C K D G C K D G C K D
UniformS 90 100 30 100 100 90 0 80 100 40 100 100
AH 80 70 10 90 80 90 0 70 90 90 100 100
NormalS 80 60 0 90 80 60 0 80 90 80 80 100
AH 90 0 0 90 90 0 0 60 100 100 100 100
ExponentialS 80 90 60 100 90 100 60 90 90 80 30 90
AH 80 90 60 90 90 100 40 90 100 90 90 100
Exp. FlippedS 80 80 60 100 90 90 30 80 100 50 80 100
AH 70 70 50 70 80 80 20 80 100 100 90 100
Bimodal GMS 90 80 0 100 90 80 0 80 100 70 100 90
AH 80 80 0 80 90 80 0 70 100 90 100 90
ArcsineS 80 100 40 90 100 100 20 80 100 60 90 100
AH 90 70 50 90 100 60 20 80 100 60 90 100
LorentzianS 80 0 50 100 70 0 20 80 80 100 80 90
AH 80 0 40 100 80 0 10 80 100 100 80 100
RayleighS 80 80 10 100 90 90 0 70 100 80 100 100
AH 80 60 10 90 90 80 0 70 100 90 100 100
Ray. FlippedS 70 60 60 100 80 60 30 70 90 70 90 90
AH 80 0 60 100 100 0 20 80 100 100 90 100
In contrast, Gemini-2.5-flash and Deepseek-v3.2-chat demonstrate high distributional invariance, forming
broad, consistent polygons in Figure 6. Their performance remains robust (consistently >80 –90%) regardless
of whether facts are biased toward the start (Exponential), the middle (Normal), or spread evenly (Uniform).
This indicates that their attention mechanisms are significantly more resilient to the “distractor” noise inherent
in varied document structures, rendering them safer choices for processing uncurated, large-scale contexts
where the location of key evidence cannot be predicted.
25