# Guided Decoding and Its Critical Role in Retrieval-Augmented Generation

**Authors**: Özgür Uğur, Musa Yılmaz, Esra Şavirdi, Özay Ezerceli, Mahmut El Huseyni, Selva Taş, Reyhan Bayraktar

**Published**: 2025-09-08 12:51:40

**PDF URL**: [http://arxiv.org/pdf/2509.06631v1](http://arxiv.org/pdf/2509.06631v1)

## Abstract
The integration of Large Language Models (LLMs) into various applications has
driven the need for structured and reliable responses. A key challenge in
Retrieval-Augmented Generation (RAG) systems is ensuring that outputs align
with expected formats while minimizing hallucinations. This study examines the
role of guided decoding in RAG systems, comparing three methods, Outlines,
XGrammar, and LM Format Enforcer, across different multi-turn prompting setups
(0-turn, 1-turn, and 2-turn). By evaluating success rates, hallucination rates,
and output quality, we provide insights into their performance and
applicability. Our findings reveal how multi-turn interactions influence guided
decoding, uncovering unexpected performance variations that can inform method
selection for specific use cases. This work advances the understanding of
structured output generation in RAG systems, offering both theoretical insights
and practical guidance for LLM deployment.

## Full Text


<!-- PDF content starts -->

Guided Decoding and Its Critical Role in
Retrieval-Augmented Generation
Özgür U ˘gur, Musa Yılmaz, Esra ¸ Savirdi,
Özay Ezerceli, Mahmut El Huseyni, Selva Ta¸ s, Reyhan Bayraktar
Newmind AI
Istanbul, Türkiye
{ougur, myilmaz, esavirdi, oezerceli, mehussieni, stas, rbayraktar}@newmind.ai
Abstract—The integration of Large Language Models (LLMs)
into various applications has driven the need for structured
and reliable responses. A key challenge in Retrieval-Augmented
Generation (RAG) systems is ensuring that outputs align with
expected formats while minimizing hallucinations. This study
examines the role of guided decoding in RAG systems, comparing
three methods, Outlines, XGrammar, and LM Format Enforcer,
across different multi-turn prompting setups (0-turn, 1-turn, and
2-turn). By evaluating success rates, hallucination rates, and
output quality, we provide insights into their performance and
applicability. Our findings reveal how multi-turn interactions
influence guided decoding, uncovering unexpected performance
variations that can inform method selection for specific use cases.
This work advances the understanding of structured output
generation in RAG systems, offering both theoretical insights and
practical guidance for LLM deployment.
Keywords—retrieval-augmented generation, guided decoding,
large language models, structured output, outlines, xgrammar,
lm format enforcer, finite-state machines, context-free grammar,
hallucination reduction
I. INTRODUCTION
The rapid rise of Large Language Models (LLMs) has
transformed natural language processing, enabling applications
across diverse domains such as question-answering, content
generation, and conversational systems. However, a persis-
tent challenge lies in ensuring that LLM outputs adhere to
specific structural formats, a critical requirement for practical
applications like data integration, API compatibility, and au-
tomated workflows. Retrieval-Augmented Generation (RAG),
introduced by [1], enhances LLMs by incorporating external
knowledge retrieval, thereby improving factual accuracy and
contextual relevance. Although RAG addresses some limita-
tions of standalone LLMs, it does not inherently guarantee
structured output, which remain essential to meet user-defined
constraints in real-world scenarios. Recent research under-
scores this gap; authors in [2] highlight the industry’s growing
demand for user-centered, restricted LLM outputs.
To bridge this gap, guided decoding backends have
emerged as a promising solution, restricting LLM output to
predefined formats or grammars. These methods leverage tech-
niques such as finite-state machines, pushdown automata, or
character-level enforcement to ensure compliance with struc-
tural requirements. For example, authors introduces Outlines
[3], a method that employs finite-state machines for efficient
and structured text generation. Similarly, other approaches suchas XGrammar [4] and LM Format Enforcer [5] offer flexible
mechanisms to enforce complex formats like JSON or domain-
specific schemas, enhancing LLM utility in structured contexts.
II. RELATEDWORK
The development of RAG and guided decoding methods
builds on a rich foundation of research aimed at enhancing
the capabilities of LLMs. This section reviews prior studies
relevant to our investigation, focusing on RAG, the challenge
of structured outputs, and the guided decoding techniques
evaluated in this study.
Structured Outputs in LLMs.The importance of struc-
tured outputs has gained increasing attention as LLMs are de-
ployed in practical settings requiring specific formats, such as
JSON, YAML, or domain-specific schemas. Industry insights
[2] highlight that unconstrained outputs often fall short in tasks
like data processing and API integration. Therefore, enforcing
structural constraints is essential, making guided decoding a
critical area of study.
Guided Decoding Methods.Guided decoding backends
constrain LLM outputs to predefined formats, addressing the
limitations of unconstrained generation. One notable approach
is Outlines, proposed in [3], which leverages finite-state ma-
chines (FSMs) to guide text generation efficiently, ensuring
compliance with regular grammars while maintaining com-
putational scalability. This method is particularly suited for
applications that require predictable structured output.
For complex structures like context-free grammars (e.g.,
JSON or code), XGrammar uses pushdown automata to en-
force syntax, offering flexibility and precision. In contrast, LM
Format Enforcer [5] applies strict character-level constraints,
with its practical use detailed in Gat’s publicly available
implementation despite lacking a formal paper. While guided
decoding enforces strict structure, it can be computationally
intensive. To address this limitation, speculative methods such
as Ranked Speculative Decoding (RSD) improve efficiency by
using draft models and reward-based token selection, speeding
up generation without sacrificing quality, especially for long
texts [6].
Monitor-Guided Decoding for Code Completion.Mod-
ular Guided Decoding (MGD) uses static analysis to im-
prove code generation, boosting compilation rates and enabling
smaller models to outperform larger ones. It generalizes across
languages and coding constraints [7]. 979-8-3315-6655-5/25/$31.00 ©2025 IEEEarXiv:2509.06631v1  [cs.CL]  8 Sep 2025

III.GUIDEDDECODINGIMPACT ONRAG PERFORMANCE
A. Experiment Setup
We conducted experiments using a high-performance in-
ference engine powered by vLLM, withxgrammaras the
default backend for guided decoding. This setup is designed for
structured output generation, such as JSON schema or regex-
based decoding. Other supported backends includelm format
enforcerandoutlines.
Initially, vLLM v0 was employed for its compatibility
with structured decoding frameworks. vLLM v1 introduced is-
sues, restricting decoding to thexgrammar:no_fallback
mode, which generates errors with unsupported schemas. Con-
sequently, vLLM v0 remains our preferred implementation.
Future updates to vLLM v1 are expected to address these
limitations.
The experiment utilized the OpenAI-compatible server
models Qwen2.5-72B-Instruct and LLaMA-3.3-70B-Instruct.
The setup involved four stages:
1)Retrieving Relevant Documents: Using RAG to
extract query-specific context.
2)Defining Model Input: Setting structured instruc-
tions, prompts, and response formats.
3)Configuring Guided Decoding: Testing Outlines,
XGrammar, and LM Format Enforcer under identical
conditions.
4)Evaluating Multi-Turn Conversations: Assessing
performance across 0-turn, 1-turn, and 2-turn scenar-
ios.
B.MultiTurn Algorithm
Algorithm 1MultiTurn RAG Eval
1:functionMULTITURNEVAL(dataset,n)
2:chat_hist←[systemprompt]
3:forj←1tondo▷add n example turns
4:usr_ex←"rag ctx: {ctx} query: {q}"
5:asst_ex←"resp: {r} doc ids: {truth_id}"
6:chat_hist←chat_hist+ [usr_ex, asst_ex]
7:end for
8:usr_eval←"rag ctx: {ctx} query: {q}"
9:chat_hist←chat_hist+ [usr_eval]
10:model_resp←GetModelResp(chat_hist)
11:resp_ids←ExtractIDs(model_resp)▷regex
12:result←Eval(truth_id, resp_ids)
13:returnresult
14:end function
15:functionEVAL(truth_ids,resp_ids)
16:corr←[iforiinresp_idsifiintruth_ids]
17:fp←[iforiinresp_idsifinot intruth_ids]
18:tp←(|corr|>0and|fp|= 0)
19:return{
20:”success” :tp,
21:”hallucination” :|fp|>0
22:}
23:end function
The algorithm 1 describes the multi-turn RAG evaluation
process, illustrating how it operates at different levels of depth
of conversation.The algorithm takes as input a dataset of contexts, queries,
and reference document identifiers, with parameter n speci-
fying the number of exemplar turns. Forn=0, it uses only
the system prompt and the evaluation query; forn=1,2, it
prepares the corresponding exemplar exchanges demonstrating
the expected discourse patterns and the reference citation
methodology.
The framework constructs a conversation history with ex-
emplar exchanges followed by the evaluation query. For each
query, it retrieves the corresponding RAG context containing
ground truth document identifiers. The model response is
evaluated by comparing its extracted document references with
the identifiers in the RAG context. A response is considered
successful when it references at least one correct document
identifier while avoiding hallucinated references.
This framework enables systematic analysis of how conver-
sation history depth affects retrieval accuracy and hallucination
rates, providing insights into multiturn RAG system behavior.
C.Guided Decoding Methods
1)FSM-Based Outlines
This approach leverages finite-state machines (FSMs) for
efficient text generation, guaranteeing structural validity with
O(1)complexity per token. It is especially well-suited to
domains that require strict syntactic or semantic constraints,
such as legal and technical documentation.
The comments on the algorithm suggest its application in a
broader context, particularly in parsing. By applying Algorithm
1 to each string in a setVusing combined FSMs for each parse
state, it becomes possible to determine parser configurations.
These configurations include the Pushdown Automaton (PDA)
states, corresponding FSM states, and potential terminal sym-
bols. The analogy extends to using the pre-image of the PDA’s
transition map to identify PDA stack values that can read the
PDA statesq∈Qand terminal symbol setsVof a parser
configuration.
FSM Representation of Constraints.Outlines represent
regular expressions and context-free grammars (CFGs) as
finite-state machines, where states correspond to valid prefixes
of a structured sequence. The FSM tracks valid transitions,
determining which tokens can legally follow a given sequence.
This eliminates the need for exhaustive vocabulary filtering.
Efficient Vocabulary Indexing.To accelerate constraint
enforcement, Outlines precomputes a mapping from FSM
states to valid tokens. This mapping denoted asσ:Q→
P(V), enables constant-time token validity checks. Unlike
naive approaches that iterate over all vocabulary tokens per
step, Outlines retrieves valid tokens inO(1)time on average.
[8]
Token Sampling with FSM Constraints.During infer-
ence, the Outlines method modifies token sampling by apply-
ing FSM constraints to ensure structured outputs. The FSM
tracks the current state and dynamically determines the valid
token set. The next token is sampled from a constrained prob-
ability distribution, adhering to structural rules. This method is
applicable to various formats, such as floating-point numbers,
programming syntax, and structured data like JSON and XML.

2)XGrammar (Pushdown Automata-Based)
XGrammar is a high-performance engine that accelerates
LLMs by100×using precomputed token masks, a persistent
execution stack, and parallel grammar processing, supporting
real-time generation with broad compatibility.
Vocabulary Partitioning and Token Mask Optimiza-
tion:Tokens are classified as context-independent or context-
dependent, with an adaptive cache to reduce memory and speed
up validation.
Persistent Execution Stack:Manages parsing states ef-
ficiently, enabling fast branching and minimal memory over-
head.
Pushdown Automata Optimization:Improves CFG pars-
ing by inlining rules and reducing ambiguity.
Parallel Mask Generation with LLM Inference:Runs
grammar processing parallel to GPU-based inference, mini-
mizing latency.
3)LM Format Enforcer
LM Format Enforcer ensures adherence to predefined for-
mats by filtering token probabilities, allowing only compliant
tokens. It integrates with local LMs to improve reliability and
consistency. Unlike rigid methods, it offers flexible enforce-
ment that preserves the model’s formatting style, dynamically
evaluating valid token sequences to balance compliance with
autonomy, ensuring high output quality.
D.Dataset
The dataset used in this study contains metadata spanning
multiple dialogue turns. Although we report a total of 750
samples, only 507 are publicly accessible on Hugging Face
owing to privacy restrictions. The dataset can be accessed via
our Hugging Face repository1.
Table I: Dataset Overview Across Different Turns
Metric 0-Turn 1-Turn 2-Turns
Total Ref. 4909 2482 1622
Unique Ref. 3614 1955 1310
Total Samples 750 375 250
IV.RESULTS ANDDISCUSSION
We present result graphs that illustrate the differences
between the three guided decoding methods.
Our implementation addressed several challenges
specific toTurkish legal documents.The specialized
(doc_id)document_id(/doc_id)format required
precise enforcement, with all methods showing significant
improvement in handling these complex structures as
conversation turns increased. Despite the complexity of
agglutinative Turkish morphology and specialized legal
vocabulary, all guided decoding approaches maintained
high semantic quality (judge scores consistently higher than
91). Complex references in Turkish legal documents (e.g.,
"344.0321.DOR.2021_1630505603_page_623")
1https://huggingface.co/datasets/newmindai/siu-rag-datawere increasingly well-handled in multi-turn scenarios, with
false positive rates dropping dramatically from hundreds to
single digits.
Table II: E2E Generation Time per Sample (sec)
Backend LLaMA-3.3-70B-Instruct Qwen2.5-72B-Instruct
Outlines 30.642 50.766
XGrammar 30.282 50.784
LM Format Enforcer 30.534 51.468
As shown in Table II, LLaMA-3.3-70B-Instruct processes
fewer tokens per sample than Qwen2.5-72B-Instruct, which
supports larger inputs and produces more extensive outputs in
multiturn contexts. This indicates that Qwen2.5-72B-Instruct
is optimized for longer, more complex queries, whereas
LLaMA-3.3-70B-Instruct delivers faster responses on simpler
tasks. Furthermore, as multiturn complexity grows, generation
time increases proportionally. In these scenarios, LLaMA-3.3-
70B-Instruct remains more time-efficient per sample, while
Qwen2.5-72B-Instruct maintains higher throughput with larger
token sets.
Structured Output Performance Across Conversational
Scenarios.Table III and Figure 1 present a comparative
analysis of false positive rates for guided decoding methods
across 0-, 1-, and 2-turn conversational settings. In zero-
turn interactions, LM Format Enforcer (LMF) consistently
achieved the lowest false positive rates (0.49%for Qwen2.5-
72B-Instruct,3.06%for Llama-3.3-70B-Instruct), outperform-
ing Outlines and XGrammar. As conversational depth in-
creased, all methods demonstrated improved performance, with
LMF maintaining superior robustness: achieving the lowest
rates in 1-turn (0.73%and0.33%) and 2-turn (0.30%and
0.06%) contexts for Qwen2.5-72B-Instruct and Llama-3.3-
70B-Instruct, respectively. Notably, Outlines and XGrammar
exhibited marked gains in multi-turn settings, particularly with
Qwen2.5-72B-Instruct, underscoring the benefits of conversa-
tional context in reducing structural output errors.
Table III: False Positive Rates of Guided Decoding
Model Turns Outlines XGrammar LMF
Qwen2.5-72B-Instruct0-Turn 0.65% 0.61% 0.49%
1-Turn 0.32% 0.41% 0.73%
2-Turns 0.18% 0.12% 0.30%
Llama-3.3-70B-Instruct0-Turn 3.20% 3.08% 3.06%
1-Turn 0.24% 0.53% 0.33%
2-Turns 0.48% 0.31% 0.06%
Few-turn prompting significantly improved reliability, par-
ticularly in 1-turn scenarios, where explicit examples clarified
the desired output structure. Outlines and XGrammar benefited
the most, while LM Format Enforcer struggled with the added
complexity of 2-turn prompting. Among the methods, Outlines
balanced flexibility and enforcement effectively, XGrammar
offered strong performance and efficiency, and LM Format
Enforcer ensured strict structural compliance but often at the
cost of usability. Together, guided decoding and few-turn
prompting complement each other, ensuring structured and
factual outputs in RAG systems.
In a RAG scenario with10 millionchunks and100,000
unique queries, each potentially linked todistinct references,
the decoding algorithm significantly affects reference accuracy.

0-Turn 1-Turn 2-Turn020406080100Correct References (%)Qwen2.5-72B-Instruct
0-Turn 1-Turn 2-TurnLlama-3.3-70B-InstructOutlines LM Format Enforcer XGrammarFigure 1.Performance of guided decoding backends across
multi-turn scenarios
ForQwen2.5-72B-Instruct, replacingLM Format En-
forcerwithXGrammarresulted in1,600additional missed
references in thezero-turnsetting and4,000more in theone-
turncase. The performance drop suggests that XGrammar’s
format handling introduces notable degradation in single-turn
and multi-turn contexts.
Similarly, forLLaMA-3.3-70B-Instruct, XGrammar led
to134additional misses in zero-turn and2,000in one-turn,
relative to the top-performing decoding strategies such as
Outlines and LM Format Enforcer. These findings show that
in large-scale RAG systems, decoding strategy is critical for
ensuring factual consistency and reference accuracy. Default
decoding methods can fall short in high-recall tasks, leading
to grounding errors. Outlines propose enhanced control over
the output, which can substantially improve the fidelity of
references.
V. LIMITATIONS
Regex & Character Support: Outlines limited regex
support, lacking advanced features, restricts complex text
processing. Its character constraints also hinder non-ASCII
handling, reducing applicability in multilingual domains.
Generation Flexibility: Unlike LM Format Enforcer Out-
lines does not accommodate beam search or batched genera-
tion, reducing flexibility for tasks that require varied output-
sampling strategies [9]. Additionally, its partial JSON poten-
tially yielding suboptimal outputs when slight deviations are
acceptable [10].
Structured-Output Adaptability.: The absence of
optional-field support in JSON outputs further limits
adaptability in production workflows that require flexible
structured-generation methods.
Moreover, effective grammar enforcement should follow
the model’s reasoning to ensure logical outputs. While tools
like XGrammar and Outlines offer immediate syntax checks,
LM Format Enforcer lacks support for delayed enforcement
aligned with generation dynamics. XGrammar’s reliance on
manual rule specification and its computational overhead un-
derline the need for more adaptive, lightweight grammatical-
enforcement approaches.VI.CONCLUSION
Guided decoding is critical for reliable LLM deployments.
This study underscores the importance of combining structured
prompting with guided decoding to optimize RAG systems.
Integrating external retrieval with decoding strategies that en-
sure format adherence enhances factual accuracy and structural
reliability. Multi-turn prompting further improves control over
generation while maintaining consistency with application-
specific requirements. These findings highlight the need for
structured decoding and prompting to advance LLM accuracy,
usability, and integration in high-stakes applications.
This study explored the impact of decoding strategies
on reference accuracy in large language models, comparing
XGrammar with flexible methods like Outlines and LM Format
Enforcer. Experiments on Qwen2.5-72B-Instruct and LLaMA-
3.3-70B-Instruct show that adaptive strategies significantly
reducereference loss, especially in multi-turn settings.
ACKNOWLEDGMENT
This study is supported byGSI Attorney Partnership.
The authors would also like to express their gratitude for the
valuable insights and support provided throughout the research
process.
REFERENCES
[1] Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks. (2021), https://arxiv.org/abs/2005.11401
[2] Liu, M., Liu, F., Fiannaca, A., Koo, T., Dixon, L., Terry, M. & Cai,
C. “We Need Structured Output”: Towards User-centered Constraints
on Large Language Model Output.Extended Abstracts Of The CHI
Conference On Human Factors In Computing Systems. pp. 1-9 (2024,5),
http://dx.doi.org/10.1145/3613905.3650756
[3] Willard, B. & Louf, R. Efficient Guided Generation for Large Language
Models. (2023), https://arxiv.org/abs/2307.09702
[4] Dong, Y ., Ruan, C., Cai, Y ., Lai, R., Xu, Z., Zhao, Y . & Chen, T.
XGrammar: Flexible and Efficient Structured Generation Engine for
Large Language Models. (2024), https://arxiv.org/abs/2411.15100
[5] Noamgat Noamgat/LM-Format-enforcer: Enforce the output for-
mat (JSON schema, regex etc) of a language model.GitHub.,
https://github.com/noamgat/lm-format-enforcer
[6] B. Liao et al., Reward-Guided Speculative Decoding for Efficient LLM
Reasoning. 2025. [Online]. Available: https://arxiv.org/abs/2501.19324
[7] Agrawal, L. A., Svyatkovskiy, A., Sundaresan, N., & Allamanis, M.
(2023). Guiding language models of code with global context using
monitors. arXiv preprint arXiv:2306.10763.
[8] Lew, A. K., Zhi-Xuan, T., Grand, G. & Mansinghka, V . K. Sequential
Monte Carlo Steering of Large Language Models using Probabilistic
Programs.arXiv preprint arXiv:2306.03081(2023).
[9] G. P. by B. and R. Hat, “Structured Decoding in vLLM: a gentle introduc-
tion,” vLLM Blog, Jan. 14, 2025. https://blog.vllm.ai/2025/01/14/struct-
decode-intro.html
[10] Li, J., Li, J., Wang, Y ., Chang, Y ., & Wu, Y . (2025). StructFlowBench: A
Structured Flow Benchmark for Multi-turn Instruction Following. arXiv
preprint arXiv:2502.14494.