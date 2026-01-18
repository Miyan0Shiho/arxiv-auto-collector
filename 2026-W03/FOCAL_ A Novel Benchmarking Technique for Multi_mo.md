# FOCAL: A Novel Benchmarking Technique for Multi-modal Agents

**Authors**: Aditya Choudhary, Anupam Purwar

**Published**: 2026-01-12 09:46:06

**PDF URL**: [https://arxiv.org/pdf/2601.07367v1](https://arxiv.org/pdf/2601.07367v1)

## Abstract
With the recent advancements in reasoning capabilities, tool calling using MCP servers and Audio Language Models (ALMs), development and integration of multi-modal agents (with voice and text support) has come to the industry forefront. Cascading pipelines for voice agents still play a central role in the industry owing to their superior reasoning capabilities facilitated by LLMs. Although, cascading pipelines often present error propagation through the pipeline. We propose a framework, FOCAL to benchmark end-to-end reasoning, component-wise error propagation and error analysis for automated as well as human-assisted testing of multi-modal agents (voice to voice + text input). We also share two novel metrics viz. Reasoning and Semantic scores to evaluate efficacy of the agent in having meaningful conversations in voice mode.

## Full Text


<!-- PDF content starts -->

FOCAL: A Novel Benchmarking Technique for
Multi-modal Agents
Aditya Choudhary†
Sprinklr AI
Gurugram, IndiaAnupam Purwar∗†
Sprinklr AI
Gurugram, India
Abstract—With the recent advancements in reasoning capa-
bilities, tool calling using MCP servers and Audio Language
Models (ALMs), development and integration of multi-modal
agents (with voice and text support) has come to the industry
forefront. Cascading pipelines for voice agents still play a central
role in the industry owing to their superior reasoning capabilities
facilitated by LLMs. Although, cascading pipelines often present
error propagation through the pipeline. We propose a framework,
FOCAL to benchmark end-to-end reasoning, component-wise
error propagation and error analysis for automated as well as
human-assisted testing of multi-modal agents (voice to voice +
text input). We also share two novel metrics viz. Reasoning
and Semantic scores to evaluate efficacy of the agent in having
meaningful conversations in voice mode.
Index Terms—Multi-modal Agents, Voice-to-voice model, Text
to Speech (TTS), Benchmark, Retrieval Augmented Generation
(RAG).
I. INTRODUCTION
Cascading multi-modal agents which support voice input
and output, comprise of three principal components, Auto-
mated Speech Recognition (ASR), Text based agent (LLM)
built using Retrieval Augmented Generation (RAG) [1] [2]
and Text-to-Speech (TTS), each of which are susceptible
to errors of different kinds. An error committed by any of
the components propagates through the pipeline, disrupting
the expected response. There exist various benchmarks for
evaluating performance of ALMs and cascading voice-to-voice
models [3], [4]. However, these function in a broad scope in
a query-answer mode, lacking multi-turn conversations and
the particular attention needed towards evaluation of agents in
terms of adhering to a well defined agent behavior, quality
and speed of assistance, voice cloning and consistency is
not assessed in these approaches. The V oiceAssistant-Eval
benchmark [5] works in the direction of narrowing down
the scope of evaluation. However, the benchmark moves in
the direction of evaluating assistants, which are fundamen-
tally different from agents in their responses, being primarily
intended for question-answer tasks and seldom relying on
tool-calling for complex tasks requiring high-level planning.
Addressing the above gaps in existing benchmarks, our work
introduces a framework inclusive of novel metrics viz. rea-
soning and semantic to evaluate the performance of multi-
∗Corresponding Author: Anupam Purwar (e-mail: anu-
pam.aiml@gmail.com, https://anupam-purwar.github.io/page/)†Equal contributions by both authorsmodal agents(voice + text as input/output). By working with
transcribed outputs of the agent’s components (TTS and
ASR) and raw outputs from the LLMs supporting them;
evaluating voice quality and similarity to the cloning sample;
reasoning and semantic behavior of the agent, we present a
framework FOCAL grounded attentive to different facets of
a standard cascading multi-modal (voice + text input/output)
agent pipeline.
II. METHODOLOGY
A. Human-Simulator
The automated evaluation scheme models a conversation
between a user and an agent. The user mimicked by a
LLM (henceforth, referred as Human-Simulator) is prompted
to adhere to a set of guidelines. The Human-simulator is
instructed to act according to a random persona provided
to it within the prompt at runtime. The Human-simulator is
provided with data from a KnowledgeBase (KB) to generate
realistic conversation.
•Sample conversation flow indicating how users interact
with the agent to be tested
•The query/motive of the user to model the conversation
•The conversation history of user and agent messages
•Synthetic user data that may be relevant for interaction
with the agent for the given query/motive
The Human-Simulator is provided with the ability to end the
conversation at its discretion by generating a termination token
when the LLM feels it is satisfied with the agent’s response
B. Architecture
The pipeline depicted in Fig. 1a supports automated testing
as well as human involved testing for a unified framework
providing both scale and quality.
The entry into the pipeline is a seed query that is fed to
Human-Simulator to initiate the conversation with the agent.
The text generated by Human-Simulator is fed to a SOTA
TTS module built onNeuTTS[6] with support for voice
cloning to generate the input audio for the voice agent. The
voice agent’s output is fed to an ASR module governed by
whisper v3-large[7] to transcribe the agent output. The
transcription is added to the conversation history which is fed
back to the Human-Simulator to continue the conversation.
Finally, at its discretion, the Human-Simulator terminates the
conversation and stops the pipeline. The pipeline also providesarXiv:2601.07367v1  [cs.SD]  12 Jan 2026

(a)
(b)
Fig. 1: End-to-End Pipeline with conditional edges based on usage (automated vs human-involvement). Automated usage
involves use of SOTA TTS and ASR modules for interaction with the Human-Simulator. Texts are tapped into at various stages
of the pipeline to generate Ground-Truth and Implementation Transcripts (indicated in Green). The UI is displayed on the right
live communication to support evaluating metrics such as
Mean Opinion Score (MOS) demanding human feedback,
alongside automated evaluation for the other metrics. A tester
is able to provide mic inputs and hear the agent outputs within
the framework while being able to monitor relevant stages and
metrics. Further, the Human-Simulator/ Human-Evaluator may
engage in direct text based conversations alongside voice based
conversation if supported by the agent’s architecture 1a. This
enables support for complex use cases where the user may
need to engage with the agent via multi-modal input viz. text
as well as voice, thereby providing user-details or confidential
data or receiving outputs like web-links/URLs.
C. Evaluation
At each stage of conversation, the raw text outputs gen-
erated by the Human-Simulator and the agent’s LLM com-
ponent (Ground-Truth transcripts) are saved to serve as the
Ground-Truth for evaluations. The speech transcriptions by
the agent’s ASR component and by the pipeline’s ASR com-
ponent (Implementation transcripts) serve as measures of the
quality of agent’s ASR module (by comparing against Human-
Simulator’s raw output) and quality of agent’s TTS module (by
comparing against agent’s raw text output).
D. Metrics
In order to identify the error propagation through the
pipeline, we denote the following three classes of metrics,
focusing on different modalities of a voice-to-voice pipeline
•LLM as Judge: We instructgpt-4.1to analyze the
Ground-Truth and Implementation transcripts and iden-
tify the following R-E-S-T scheme
–Reasoning (R) Score: [integer on scale of 10] Scores
the agent’s reasoning capability in terms of re-
dundancy, clarity of information, quality of query
resolution. These factors play an important role for
specialized support agents
–Efficiency (E) Score: [integer] Number of messages
taken to resolve the user’s query. agent’s efficiencybecomes important in reducing the time-to-solve per
query
–Semantic (S) Score: [integer on scale of 10] Scores
the agent’s behavior in terms of politeness, under-
standing user speech and legibility of agent’s speech
through analysis of transcripts
–Tool-Calling (T) Score: [number of correct tool calls
number of ideal tool calls]
Scores the agent’s capability to resolve complex
problems into tool-calls when required with the
appropriate arguments.
•Accuracy: Evaluates ASR and TTS performance
–Word Error Rate (WER): Identifies errors committed
by agent’s ASR module by comparing the agent’s
ASR output (Implementation transcript) against
Human-Simulator’s output (Ground-Truth transcript)
and agent’s TTS module by comparing the input
received by Human-Simulator (Implementation tran-
script) and the agent’s LLM output (Ground-Truth
transcript). Lower WER indicates lesser errors com-
mitted by ASR and TTS.
–Contextual Similarity: By comparing the embeddings
of Ground-Truth and Implementation transcripts,
similarity between the semantics of the conversation
are computed using cosine similarity. Lower similar-
ity indicates higher impact of errors due to drastic
change in meaning of messages.
While WER measures the errors objectively, sometimes
these errors may be ignorable if the conversation still
retains its meaning despite these errors. Hence, contextual
similarity emerges as the more important metric that
drives the user experience while engaging with a agent.
•Vocal Quality: Evaluates the performance of agent’s TTS
and the user experience in interacting with the agent
–V oice Similarity: Utilizingwespeaker’s [8] em-
beddings for the agent’s output and the sample given
to it, we identify how well the agent is able to clone a
particular voice by calculating cosine similarity with
the sample. The similarity score undergoes a linear

transformation on 0-1 scale.
–V oice Consistency: Measures average and standard
deviation of all pairwise voice similarity scores be-
tween all agent utterances. Higher the score, the more
consistent is the agent’s voice among the different
stages in a multi-turn conversation.
–MOS: Evaluating MOS score is a fairly cost and la-
bor intensive process. In order to provide a synthetic
estimate for MOS we utilizeUTMOSv2[9] to obtain
an estimate on a scale of 1-5.
V oice Consistency emerges as an important metric con-
tributing to user experience as it measures whether the
agent’s voice is consistent across multi-turn conversa-
tions. V oice Similarity and Quality measure whether the
agent is able to provide the user with a high-fidelity
experience aligning with the voice sample intended
E. Demo Setup
To provide an interactive experience of the agent evaluation
pipeline, we have a UI (Fig. 1b) hosted on cloud through which
the users may interact with the agent and view the evaluations
in real-time. The Demo setup would comprise of two monitors
exhibiting the UI and a presentation for the pipeline. The demo
will be accessible to the visitors to converse with a agent and
run the pipeline in automated manner and observe the Human-
Simulator carry out the conversations
III. RESULTS
Developing on the work presented in [10], we utilize our
framework to benchmark a cascading voice agent using a RAG
based Shopping Support Agent built usinggpt-4o-mini.
The Human-Simulator (another LLM) is given seed queries
in form of different customer journeys such as order tracking
and finding nearest store. The RAG based agent is provided
with a knowledge in the form of a database containing relevant
information to address various customer journeys.
At runtime, the Human-Simulator is provided with a random
personality from a collection. The personality is enriched by
fetching relevant data from the database and injecting it to the
query in order to make the conversations coherent.
Different evaluation metrics viz. Mean Opinion Score
(MOS), V oice quality and Accuracy for 6 different customer
journeys are documented in Table I for the voice-to-voice
architecture experimented in this work.
IV. CONCLUSION
The space for evaluating multi-modal agents is gaining
importance day by day with advent of several technologies
that may be integrated in a multitude of ways to yield several
different architectures. Hence, it becomes important to evaluate
the performance end-to-end while also focusing on the inter-
mediate components to identify error propagation. Our work
presents a framework to handle the multi-faceted evaluations
for a multi-modal voice agent. We have evaluated FOCAL
framework using our prior published voice+text agents [10]
and as part of Demo, we plan to give a live demo of FOCAL
framework in action.Future Work
The power of the agents arises from the tools provided
to them which enables them to interact with the real-world.
However, it also raises security concerns due to vulnerabilities
of the components of agent. We next envisage to evaluating
the safety of agents through adversarial testing.
V. ACKNOWLEDGEMENT
Authors acknowledge Yoginkumar Patel and Amitabh Misra
for their encouragement to drive innovation through research.
The authors also thank the Devops Engineering team at
Sprinklr for help in setting up the infrastructure related to
FOCAL benchmark prototype.
REFERENCES
[1] Anupam Purwar and Rahul Sundar. Keyword augmented retrieval: Novel
framework for information retrieval integrated with speech interface. In
Proceedings of the Third International Conference on AI-ML Systems,
AIMLSystems ’23, New York, NY , USA, 2024. Association for Com-
puting Machinery.
[2] Kush Juvekar and Anupam Purwar. Introducing a new hyper-parameter
for rag: Context window utilization, 2024.
[3] Y . Chen, X. Yue, Chen Zhang, X. Gao, Robby T. Tan, and H. Li.
V oiceBench: Benchmarking LLM-Based V oice Assistants, 2024.
[4] H Liu, Y Wang, Z Cheng, R Wu, Q Gu, Y . Wang, and Yu Wang. V o-
calBench: Benchmarking the V ocal Conversational Abilities for Speech
Interaction Models, 2025.
[5] Ke Wang, H. Ren, Zimu Lu, M. Zhan, and H. Li. V oiceAssistant-Eval:
Benchmarking AI Assistants across Listening, Speaking, and Viewing,
2025.
[6] Neuphonic: Neutts-air, 2025.
[7] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine
McLeavey, and Ilya Sutskever. Robust Speech Recognition via Large-
Scale Weak Supervision, 2022.
[8] H. Wang, Shuai Liang, C.and Wang, Z. Chen, Binbin Zhang, Xu Xiang,
Yanlei Deng, and Yanmin Qian. Wespeaker: A research and production
oriented speaker embedding learning toolkit. InIEEE International
Conference on Acoustics, Speech and Signal Processing, 2023.
[9] K. Baba, W. Nakata, Y . Saito, and Hiroshi S. The T05 System for
The V oiceMOS Challenge 2024: Transfer Learning from Deep Image
Classifier to Naturalness MOS Prediction of High-Quality Synthetic
Speech. InIEEE Spoken Language Technology Workshop (SLT), 2024.
[10] i-LA V A: Insights on Low Latency V oice-2-V oice Architecture for
Agents.

TABLE I: Evaluation of a cascading RAG based Shopping Agent under different modalities, (see Appendix V)
LLM Judge Accuracy V oice Quality
Customer Journey Reasoning Semantic Tool-Calling Similarity WER↓ Similarity MOS Consistency
Store Locator6 7 0/1 0.9513 0.1875 0.7037 2.778 0.6802±0.0
7 8 1/1 0.9194 0.1303 0.6894 2.6431 0.6333±0.0369
Damaged Items8 7 1/1 0.9435 0.0788 0.7325 3.006 0.7189±0.0371
10 9 1/1 0.9440 0.0390 0.7610 3.379 0.7984±0.0561
Payment Issues9 9 1/1 0.9372 0.0873 0.7292 2.980 0.7863±0.0449
10 9 1/1 0.9388 0.0752 0.7204 3.132 0.6894±0.0498
Track Order10 9 1/1 0.9465 0.0559 0.7351 2.876 0.7315±0.0654
10 9 1/1 0.9442 0.0502 0.6786 2.608 0.7556±0.0235
Return Order8 9 1/1 0.9273 0.0615 0.7331 2.913 0.7468±0.0493
8 8 1/1 0.9452 0.1785 0.7645 2.432 0.7766±0.0849
Cancel Order10 9 1/1 0.9796 0.0625 0.6723 2.930 0.7169±0.0459
9 9 1/1 0.9735 0.0625 0.6989 2.755 0.6733±0.0437

APPENDIX
I. CONVERSATIONS
#1
USER: Hi, can you help me find a store nearby?
ASSISTANT: Could you please provide me with your current location or the area you’re in so I
can find a store nearby for you?
USER: My pincode is 66764.
ASSISTANT: The nearest store for your pincode 66764 is located in New York at 4758 Ben Dale.
You can find it on the map [here](https://maps.app.goo.gl/aGkvAvRduPymoinGA).
#2
USER: Hi, can you help me find a store nearby?
ASSISTANT: Could you please provide me with your current location or a specific area where you
would like to find a store?
USER: My pincode is 27368.
ASSISTANT: The nearest store to your pincode 27368 is Central Park, located at 52448 Gleichner
Oval. You can find the location [here](https://maps.app.goo.gl/ZGxzsxcj44tPMAZn7).
USER: Thanks for the info! That’s all I needed.
ASSISTANT: You’re welcome! If you have any more questions in the future, feel free to ask. Have
a great day!
#3
USER: Hi, I need to replace a damaged item.
ASSISTANT: Could you please provide more details about the damaged item? Specifically, I’d like
to know what item it is and any relevant information regarding its replacement.
USER: The order ID is 3.
ASSISTANT: The order ID 3 corresponds to a Monitor, which is currently processing and has a
delivery date of June 27, 2025. Please contact customer service for further assistance with
the replacement process.
USER: Okay, I’ll contact customer service. Thanks!
ASSISTANT: You’re welcome! If you have any other questions or need further assistance, feel
free to ask. Good luck with your replacement!
#4
USER: Hi, I need to replace a damaged item.
ASSISTANT: Could you please provide me with more details about the damaged item you’d like to
replace? This includes the item name, any order numbers, and the nature of the damage.
USER: The order ID is 3.
ASSISTANT: The order ID 3 corresponds to a Monitor, which is currently in processing status.
The delivery date is set for 2025-06-27. Please let me know how you would like to proceed
with the replacement.
USER: I would like to proceed with the replacement.
ASSISTANT: I recommend contacting customer service directly to initiate the replacement process
for your monitor with order ID 3. They will guide you through the necessary steps and
provide any additional instructions. If you need further assistance, feel free to ask!
USER: Thank you, that’s all I needed.

ASSISTANT: You’re welcome! If you have any more questions in the future or need assistance with
anything else, feel free to reach out. Have a great day!
#5
USER: Hi, I’m having an issue with payment.
ASSISTANT: Could you provide more details about the payment issue you’re experiencing? This
information will help me assist you better.
USER: The payment isn’t going through for order ID 2.
ASSISTANT: The payment status for order ID 2 (Mouse) is "Failed." You may want to check your
payment details or try a different payment method.
USER: Thanks, I’ll try that.
ASSISTANT: You’re welcome! If you have any more questions or need further assistance, feel free
to ask. Good luck with your payment!
#6
USER: Hi, I’m having an issue with payment.
ASSISTANT: Could you please provide more details about the payment issue you’re experiencing?
This will help me assist you better.
USER: The payment for order ID 5 isn’t going through.
ASSISTANT: The payment status for order ID 5 (Printer) is marked as "Failed." You may want to
check your payment information or try again.
USER: Okay, I’ll check my payment info. Thanks!
ASSISTANT: You’re welcome! If you need any further assistance or have more questions, feel free
to ask. Good luck with your payment!
#7
USER: Hi, I need to track my order.
ASSISTANT: Could you please provide your order number or any other details related to your
order? This will help me assist you better.
USER: The order ID is 1.
ASSISTANT: Your order ID 1 is for a Keyboard, ordered by Terrell F. The current status is "
processing," and the expected delivery date is June 25, 2025.
USER: Thanks for the update! That’s all I needed.
ASSISTANT: You’re welcome! If you have any more questions or need assistance in the future,
feel free to ask. Have a great day!
#8
USER: Hi, I need to track my order.
ASSISTANT: Could you please provide me with your order number or any other details related to
your order? This will help me assist you better in tracking it.
USER: The order ID is 3.
ASSISTANT: Your order ID 3 for the Monitor is currently in processing status and is expected to
be delivered on June 27, 2025.
USER: Thanks for the update!
ASSISTANT: You’re welcome! If you have any more questions or need further assistance, feel free
to ask. Happy to help!
#9
USER: Hi, I need to return an order.
ASSISTANT: Can you please provide me with the order number or any specific details about the
order you would like to return? This will help me assist you better.
USER: The order ID is 5.
ASSISTANT: Your order ID 5 is for a Printer, delivered on June 29, 2025. You can return it
until July 9, 2025. For more information on the return process, you can visit the store at
Central Park using this link: [Map Link](https://maps.app.goo.gl/ZGxzsxcj44tPMAZn7).
USER: Thanks for the info! That’s all I needed.
ASSISTANT: You’re welcome! If you have any more questions in the future or need further
assistance, feel free to ask. Have a great day!
#10
USER: Hi, I need to return an order.

ASSISTANT: Could you please provide me with more details about the order you want to return,
such as the order number, the reason for the return, and any specific policies or
conditions you are aware of? This will help me assist you better.
USER: The order ID is 5.
ASSISTANT: You can return Order ID 5, which is a Printer purchased from Central Park. The
delivery date was June 29, 2025, and you have until July 9, 2025, to initiate the return.
For more information, you can visit the return location [here](https://maps.app.goo.gl/
ZGxzsxcj44tPMAZn7).
USER: Thanks for the info. That’s all I needed.
ASSISTANT: You’re welcome! If you have any more questions in the future or need further
assistance, feel free to reach out. Have a great day!
#11
USER: Hey, I need to cancel an order.
ASSISTANT: Could you please provide me with your order number or any additional details related
to your order so I can assist you with the cancellation?
USER: The order ID is 1.
ASSISTANT: Unfortunately, you cannot cancel order 1 for the Keyboard as it is not eligible for
cancellation.
USER: Okay, thanks for letting me know.
ASSISTANT: You’re welcome! If you have any more questions or need further assistance, feel free
to ask.
#12
USER: Hey, I need to cancel an order.
ASSISTANT: Please provide me with the details of the order you’d like to cancel, such as the
order number or the item name.
USER: The order ID is 4.
ASSISTANT: The order with ID 4 is for Headphones, but it is not eligible for cancellation. You
will not be able to cancel this order.
USER: Okay, thanks for letting me know.
ASSISTANT: You’re welcome! If you have any more questions or need further assistance, feel free
to ask.
II. LLMASJUDGESCORES
#1
Reasoning Score: 6
Reasoning Reason: The agent provided a store location, but there are several issues: the
pincode in the response is incorrect (66064 instead of 66764), the address is misspelled (’
Bendale’ instead of ’Ben Dale’), and the map link is garbled and not the same as the ground
-truth. The agent also included an unclear phrase at the end (’I’ve already upymoingga’),
which is not present in the ground-truth and adds confusion.
Semantic Score: 7
Semantic Reason: The conversation flow is mostly correct: the agent asks for the user’s
location and responds with a store address. However, the unclear phrase at the end and the
incorrect information reduce the professionalism and clarity of the flow.
Tool Score: 0/1
Tool Reason: The agent should have used a store locator tool with the correct pincode (66764)
as input. Instead, the response appears to hallucinate the data, providing an incorrect
pincode and a malformed map link.
Evaluation Summary: The agent followed the general structure of the conversation but made
critical errors in the details: the pincode, address, and map link were all incorrect or
malformed. There was also an unclear and unnecessary phrase at the end. The agent needs to
ensure accuracy and clarity in responses, especially when providing location-based
information.
#2
Reasoning Score: 7
Reasoning Reason: The Implementation Transcript mostly aligns with the ground-truth but
contains some notable errors: the address is transcribed incorrectly (’Glide Mare Oval’
instead of ’Gleichner Oval’), the pincode is mispronounced as ’PINCO 270368’ instead of ’
pincode 27368’, and the location link is missing. These issues result in incomplete and
potentially confusing information for the customer.

Semantic Score: 8
Semantic Reason: The conversation flow is polite and follows the expected structure, but there
are minor issues with the agent’s closing statement (missing exclamation mark and slightly
less warmth) and the mispronunciation of ’pincode’ as ’PINCO’.
Tool Score: 1/1
Tool Reason: The correct tool (store locator) was used to find the nearest store based on the
pincode provided. There is no evidence of hallucinated data, and the tool was used
appropriately, though the output was not relayed perfectly.
Evaluation Summary: The agent was polite and prompt, asking for the necessary information and
providing a relevant response. However, there were transcription errors in the address and
pincode, and the location link was omitted, making the response less helpful than the
ground-truth. The tool usage was correct, but the delivery of information could be improved
for accuracy and completeness.
#3
Reasoning Score: 8
Reasoning Reason: The Implementation Transcript conversation mostly aligns with the ground-
truth, but there are minor issues: the delivery date is incorrectly stated as ’June 27,
2000-2025’ instead of ’June 27, 2025’, and the final message contains a typo (’rock
placement’ instead of ’replacement’).
Semantic Score: 7
Semantic Reason: The flow is generally polite and follows the ground-truth, but the closing
message contains an inappropriate phrase (’rock placement’), which could confuse the
customer and detracts from professionalism.
Tool Score: 1/1
Tool Reason: The agent correctly references the order details, which implies the correct tool
was used to fetch order information. No hallucination or missing tool call is evident.
Evaluation Summary: The agent handled the query efficiently and provided the necessary
information. However, there were minor transcription or phrasing errors, particularly with
the delivery date and the closing message, which could impact customer understanding and
satisfaction. Overall, the conversation was helpful and polite, but attention to detail in
responses could be improved.
#4
Reasoning Score: 10
Reasoning Reason: The Implementation Transcript conversation closely matches the ground-truth
in terms of information provided, steps followed, and responses to the customer’s queries.
All necessary information fields are present and there is no redundancy or skipping of
steps.
Semantic Score: 9
Semantic Reason: The flow of the conversation is polite, clear, and follows the ground-truth
closely. The only minor difference is the lack of exclamation marks and slightly less
enthusiastic closing statements, but these are negligible and expected in transcription.
Tool Score: 1/1
Tool Reason: The agent correctly references the order details after receiving the order ID,
which implies a correct tool call to fetch order information. No hallucinated data or
missing tool usage is observed.
Evaluation Summary: The agent handled the conversation efficiently and provided all necessary
information without redundancy or skipping steps. The tone was polite and professional,
closely matching the ground-truth. Minor differences in punctuation and enthusiasm do not
impact the overall quality.
#5
Reasoning Score: 9
Reasoning Reason: The Implementation Transcript conversation closely aligns with the ground-
truth. The only minor discrepancy is the agent’s response: ’order IV2 mouse’ instead of ’
order ID 2 (Mouse)’. This appears to be a transcription or pronunciation error, but all
necessary information is present and no steps are skipped or added.
Semantic Score: 9
Semantic Reason: The flow is polite and matches the ground-truth, with proper salutations and
closure. The only minor issue is the lack of an exclamation mark in the final response,
which is negligible for flow.
Tool Score: 1/1
Tool Reason: The agent correctly checked the payment status for the provided order ID, as
required by the ground-truth. No hallucination or missing tool calls.
Evaluation Summary: The agent provided accurate and helpful information, maintained a polite
and empathetic tone, and followed the ideal conversation flow. The only minor weakness was

a transcription error in the order ID and product name, but this did not impact the overall
quality or understanding of the conversation.
#6
Reasoning Score: 10
Reasoning Reason: The Implementation Transcript conversation closely matches the ground-truth.
All necessary information fields are present, and there is no redundancy, skipping of steps
, or lack of information.
Semantic Score: 9
Semantic Reason: The flow is nearly identical to the ground-truth, with only minor differences
in punctuation and a missing exclamation mark in the closing salutation. Politeness and
promptness are maintained throughout.
Tool Score: 1/1
Tool Reason: The agent correctly checked and reported the payment status for the specified
order, as required by the ground-truth. No unnecessary or missing tool calls.
Evaluation Summary: The agent provided accurate and complete information, maintained a polite
and helpful tone, and followed the ideal conversation flow. Minor differences in
punctuation do not affect the quality of support. Overall, the conversation is well-aligned
with the ground-truth.
#7
Reasoning Score: 10
Reasoning Reason: The Implementation Transcript conversation closely matches the ground-truth.
All key information fields (order item, customer name, status, and expected delivery date)
are present. There is no redundancy or skipped steps.
Semantic Score: 9
Semantic Reason: The flow is polite and professional, with proper salutations and closing.
There is a minor omission of an exclamation mark in the closing, but this is negligible and
expected in transcriptions.
Tool Score: 1/1
Tool Reason: The agent correctly requested the order number and provided the order status,
indicating the correct use of the order tracking tool.
Evaluation Summary: The Implementation Transcript transcript is highly aligned with the ground-
truth, providing all necessary information in a clear and concise manner. The conversation
flow is smooth and professional, with only minor, inconsequential differences in
punctuation. Tool usage is appropriate and accurate.
#8
Reasoning Score: 10
Reasoning Reason: The Implementation Transcript conversation is fully aligned with the ground-
truth. All information fields are present, and there is no redundancy, skipping of steps,
or lack of information.
Semantic Score: 9
Semantic Reason: The flow is polite and matches the ground-truth almost exactly. The only minor
difference is the omission of an exclamation mark in the final response, which is
negligible and expected in transcriptions.
Tool Score: 1/1
Tool Reason: The agent correctly requested the order ID and presumably used the appropriate
tool to fetch the order status, as in the ground-truth.
Evaluation Summary: The Implementation Transcript conversation is concise, polite, and provides
all necessary information. The agent follows the ideal flow and tool usage, with only a
negligible difference in punctuation. Overall, the conversation is highly effective and
customer-friendly.
#9
Reasoning Score: 8
Reasoning Reason: The Implementation Transcript conversation closely aligns with the ground-
truth in terms of information provided about the order, delivery date, and return window.
However, it omits the specific map link for the store, which is present in the ground-truth
. The product name is also not capitalized (’printer’ vs ’Printer’), which is a minor
detail.
Semantic Score: 9
Semantic Reason: The flow is polite, clear, and matches the ground-truth in structure and tone.
The only minor deviation is the omission of the exclamation mark in the final agent
response, and the slightly less enthusiastic closing, but these are negligible.
Tool Score: 1/1
Tool Reason: The agent correctly used the necessary tool to fetch order details and return

information. There is no evidence of hallucinated data or missing tool calls.
Evaluation Summary: The Implementation Transcript conversation is strong in terms of accuracy,
politeness, and completeness. The only notable weakness is the omission of the specific map
link for the store, which would have provided a more complete answer. Otherwise, the agent
handled the query efficiently and courteously.
#10
Reasoning Score: 8
Reasoning Reason: The Implementation Transcript conversation covers all the key information
fields: order details, delivery date, return deadline, and a return location link. However,
the return location URL is pronounced in a confusing and incorrect manner (’dot, do, dot,
do, dot, G-L, slash, Z-G-A-Z-S-X-E-J-44. It emails in seven.’) instead of providing a clear
or clickable link as in the ground-truth. The rest of the information is accurate and
complete.
Semantic Score: 8
Semantic Reason: The conversation flow is polite and follows the ground-truth closely, with
appropriate greetings and closure. However, the agent’s pronunciation of the return
location link is awkward and unclear, which could confuse the customer. The rest of the
flow is smooth and professional.
Tool Score: 1/1
Tool Reason: The agent correctly retrieves the order and return information based on the
provided order ID, as required. There is no evidence of hallucinated data or missing tool
calls.
Evaluation Summary: The agent provided accurate and complete information regarding the return
process, including order details and deadlines. The main weakness is the unclear and
incorrect pronunciation of the return location link, which could hinder customer
understanding. Otherwise, the conversation is polite, concise, and follows the ideal flow.
#11
Reasoning Score: 10
Reasoning Reason: The Implementation Transcript conversation closely matches the ground-truth.
All information fields are present, and there is no redundancy, skipping of steps, or lack
of information.
Semantic Score: 9
Semantic Reason: The flow is polite and matches the ground-truth, but there is a minor
difference in the phrasing of the product name (’keyboard’ vs ’Keyboard’) and a missing
exclamation mark in the final response. However, these are minor and do not affect the
overall flow.
Tool Score: 1/1
Tool Reason: The agent correctly requested the order number as in the ground-truth, which is
the only tool-like action required in this scenario.
Evaluation Summary: The Implementation Transcript conversation is well-aligned with the ground-
truth, providing accurate and complete information with a polite and helpful tone. Minor
differences in capitalization and punctuation do not impact the quality of support.
#12
Reasoning Score: 9
Reasoning Reason: The Implementation Transcript conversation is almost fully aligned with the
ground-truth. The only minor discrepancy is a typographical error where ’ID 4’ is
pronounced as ’IB4’, which could cause confusion. Otherwise, all information fields and
steps are present.
Semantic Score: 9
Semantic Reason: The flow is polite and matches the ground-truth, with proper salutations and
closure. The only minor issue is the typo in the order ID, which slightly affects clarity.
Tool Score: 1/1
Tool Reason: The agent correctly requested the order details and provided the cancellation
eligibility as expected. There is no evidence of hallucinated data or missed tool calls.
Evaluation Summary: The agent handled the conversation well, providing accurate and complete
information with a polite and helpful tone. The only weakness is a minor transcription
error in the order ID (’IB4’ instead of ’ID 4’), but this did not significantly impact the
overall support quality.