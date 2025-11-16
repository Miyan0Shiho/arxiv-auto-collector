# MINDS: A Cross-cultural Dialogue Corpus for Social Norm Classification and Adherence Detection

**Authors**: Pritish Sahu, Anirudh Som, Dimitra Vergyri, Ajay Divakaran

**Published**: 2025-11-13 03:33:39

**PDF URL**: [https://arxiv.org/pdf/2511.09918v1](https://arxiv.org/pdf/2511.09918v1)

## Abstract
Social norms are implicit, culturally grounded expectations that guide interpersonal communication. Unlike factual commonsense, norm reasoning is subjective, context-dependent, and varies across cultures, posing challenges for computational models. Prior works provide valuable normative annotations but mostly target isolated utterances or synthetic dialogues, limiting their ability to capture the fluid, multi-turn nature of real-world conversations. In this work, we present Norm-RAG, a retrieval-augmented, agentic framework for nuanced social norm inference in multi-turn dialogues. Norm-RAG models utterance-level attributes including communicative intent, speaker roles, interpersonal framing, and linguistic cues and grounds them in structured normative documentation retrieved via a novel Semantic Chunking approach. This enables interpretable and context-aware reasoning about norm adherence and violation across multilingual dialogues. We further introduce MINDS (Multilingual Interactions with Norm-Driven Speech), a bilingual dataset comprising 31 multi-turn Mandarin-English and Spanish-English conversations. Each turn is annotated for norm category and adherence status using multi-annotator consensus, reflecting cross-cultural and realistic norm expression. Our experiments show that Norm-RAG improves norm detection and generalization, demonstrates improved performance for culturally adaptive and socially intelligent dialogue systems.

## Full Text


<!-- PDF content starts -->

MINDS: A Cross-cultural Dialogue Corpus for
Social Norm Classification and Adherence Detection
Pritish Sahu*, Anirudh Som*, Dimitra Vergyri, Ajay Divakaran
SRI International
{pritish.sahu, contact.anirudh.som}@gmail.com, {dimitra.vergyri, ajay.divakaran}@sri.com
Abstract
Social norms are implicit, culturally grounded
expectations that guide interpersonal commu-
nication. Unlike factual commonsense, norm
reasoning is subjective, context-dependent, and
varies across cultures, posing challenges for
computational models. Prior works provide
valuable normative annotations but mostly tar-
get isolated utterances or synthetic dialogues,
limiting their ability to capture the fluid, multi-
turn nature of real-world conversations. In
this work, we presentNorm-RAG, a retrieval-
augmented, agentic framework for nuanced
social norm inference in multi-turn dialogues.
Norm-RAG models utterance-level attributes
including communicative intent, speaker roles,
interpersonal framing, and linguistic cues and
grounds them in structured normative documen-
tation retrieved via a novel Semantic Chunk-
ing approach. This enables interpretable and
context-aware reasoning about norm adherence
and violation across multilingual dialogues. We
further introduceMINDS(Multilingual Inter-
actions with Norm-Driven Speech), a bilingual
dataset comprising 31 multi-turn Mandarin-
English and Spanish-English conversations.
Each turn is annotated for norm category and
adherence status using multi-annotator consen-
sus, reflecting cross-cultural and realistic norm
expression. Our experiments show that Norm-
RAG improves norm detection and generaliza-
tion, demonstrates improved performance for
culturally adaptive and socially intelligent dia-
logue systems.
1 Introduction
Social norms are culturally embedded, often im-
plicit expectations that shape how individuals in-
teract in society, especially in interpersonal dia-
logues. These norms whether, behavioral conven-
tions, moral obligations, or expectations of po-
liteness guide acceptable conduct and influence
*Equal contributionboth verbal and non-verbal communication (Sherif,
1936; Haidt, 2012; Schwartz et al., 2012). As
norms can vary significantly across cultures (Trian-
dis, 1994; Arieli, 1964), modeling them in com-
putational systems demands reasoning beyond the
literal meaning of utterances. Unlike factual com-
monsense, social norm reasoning involves subjec-
tivity, context, and cultural nuances, making it sig-
nificantly more ambiguous and under-determined.
Recent works, however, have made strides in
codifying normative knowledge through structured
resources such as SocialChem (Forbes et al., 2020),
NormSage (Fung et al., 2022), NormDial (Li et al.,
2023), SocialDial (Zhan et al., 2023) and RENOVI
(Zhan et al., 2024), which annotate social norms
in descriptive scenarios or conversations. These
datasets highlight the importance of understanding
values, intents, justifications, and social expecta-
tions. However, they fall short in modeling the
dynamic, multi-turn nature of conversations where
shifts in intent, emotional alignment, or interper-
sonal sensitivity can dramatically affect norm in-
terpretation. For example, identifying whether a
speaker’s disagreement is socially acceptable may
depend on factors such as power dynamics, tone,
or context established across earlier turns. Exist-
ing resources often lack fine-grained annotations
for cues, interpersonal relationships, or latent in-
tentions which are critical signals for robust social
norm understanding.
To address these limitations we proposeNorm-
RAG, a retrieval-augmented generation based agen-
tic framework for social norm inference in dialogue.
It models utterance-level features such as intent,
cue, interest, and role interplay, enabling a more
nuanced detection of implicit and culturally sen-
sitive social norms in natural conversation. Fur-
thermore, it decomposes norm behavior into the
following pragmatically grounded attributes to en-
hance the interpretability and accuracy of norm
detection -Communicative Intent,InterpersonalarXiv:2511.09918v1  [cs.CL]  13 Nov 2025

Framing,Linguistic Features, andContextual Trig-
gers and Constraints. By integrating these struc-
tured representations with LLM-based reasoning
and dynamic retrieval of relevant norm documenta-
tion, we support both norm classification and norm
adherence/violation assessment in multilingual con-
versation settings.
We also releaseMINDS, short forMultilingual
Interactions withNorm-DrivenSpeech, a novel
bilingual dataset consisting of 31 annotated multi-
turn dialogue sessions across Mandarin-English
and Spanish-English pairs. Each turn is labeled
for norm category and adherence/violation status
labels, with multi-annotator consensus to ensure
quality and consistency. Unlike prior datasets that
rely on static prompts or synthetic conversations,
ours captures natural, two-party interactions, en-
abling more realistic modeling of culturally embed-
ded norm violations and their detection.
Our key contributions are summarized as follows:
•We developedNorm-RAG, a novel agentic
architecture that models social norms through
retrieval-augmented generation, leveraging
feedback from prior utterances and structured
dialog context for turn-level inference.
•We introduce theMINDS corpus, a bilingual,
multi-annotated dialogue dataset covering
Spanish-English and Mandarin-English con-
versations, annotated for social norm type and
adherence/violation status, reflecting cross-
cultural, real-world interactions.
•We formulate norm classification using four
interpretable, latent dimensions:Intent,Fram-
ing,Linguistic Features, andConstraints,
moving beyond surface-level cue detection.
•We present a novelSemantic Chunkingtech-
nique for norm document retrieval, replac-
ing heuristic keyword matching with context-
aware semantic segmentation to accurately
extract applicable normative guidance.
•We benchmark and analyze retrieval-based
and generative approaches across various con-
figurations, demonstrating improved norm de-
tection performance and generalizability.
2 Related Work
Early efforts in modeling social norms computa-
tionally have centered around static textual contexts
(Ziems et al., 2023; Sap et al., 2019; Rashkin et al.,
2018; Emelin et al., 2020; Jiang et al., 2021; Kimet al., 2022; Gu et al., 2021; Ziems et al., 2022; CH-
Wang et al., 2023). Social Chemistry 101 (Forbes
et al., 2020) introduced Rules-of-Thumb (RoTs)
which are defined as free-text normative statements
tied to situational prompts annotated with categor-
ical labels capturing legality, moral foundations,
and cultural expectations. While this large-scale
resource enabled pretraining norm-aware language
models, its static, monologic format restricts ap-
plication to real-time dialogic settings. To over-
come the lack of interactive context, NormSage
(Fung et al., 2022) proposed a zero-shot prompting
method for extracting culture-specific norms from
dialogues across languages using GPT-3, creating
the NormsKB knowledge base. Though NormSage
allows dynamic norm discovery and cross-cultural
applicability, it lacks annotations for turn-level ad-
herence or violation, thereby limiting its utility in
norm-tracking tasks.
NormDial (Li et al., 2023) advanced the field
by annotating each dialogue turn with adherence,
violation, or irrelevance labels, using human-in-
the-loop generation of synthetic conversations
grounded in American and Chinese norms. How-
ever, its reliance on synthetic data and predefined
norm templates limits its coverage of spontaneous
and organically evolving dialogues. SocialDial
(Zhan et al., 2023) is a large-scale, monocultural
resource centered on a Chinese ontology of social
norms (5 categories, 14 subcategories). Its evalua-
tion tasks involve predicting dialogue-level social
factors (e.g., distance, relation, location, formal-
ity, topic) and detecting norm violations within
Chinese cultural contexts. In contrast, our work
introduces a cross-cultural, bilingual dataset span-
ning Mandarin–English and Spanish–English con-
versations derived from real conversational data,
annotated not only for norm adherence but also
for underlying speaker-level features such as intent,
cue, and interpersonal alignment. Furthermore, we
propose a dual-task framework that performs la-
tent norm discovery alongside turn-level adherence
classification, without assuming access to prede-
fined norm statements. This approach offers a more
holistic and dynamic modeling of social norms in
conversation, addressing key limitations in prior
datasets and moving closer to deployable, socially
intelligent dialog systems.
3 MINDS Corpus
Unlike static rule-sets, norms are context-sensitive,
adaptive, and frequently nuanced by linguistic, in-

PersuasionMandarin Norm Categories & Distribution Spanish Norm Categories & Distribution
Request
Criticism
Greeting
Apology
AdmirationThanks
Refusing Request
Taking Leave
Finalize Deal
400 350 300 250 200 150 100 50 0 50403 (22.1%)
239 (13.1%)
78 (4.3%)
104 (5.7%)
63 (3.5%)
53 (2.9%)
51 (2.8%)
45 (2.5%)45 (2.5%)
33 (1.8%)
10 (0.5%)16 (0.9%)
16 (0.9%)
1 (0.1%)
1 (0.1%)
1 (0.1%)
2 (0.1%)5 (0.3%)
6 (0.3%)
5 (0.3%)
Adherence Violation
150 100 50 0 50Request Info
Disagreement
Criticism
Greeting
Admiration
Refusing RequestThanks
Granting Request
Apology
Finalize Deal182 (10.0%)
62 (3.4%)
39 (2.1%)
76 (4.2%)
35 (1.9%)
42 (2.3%)
16 (0.9%)
23 (1.3%)
6 (0.3%)
9 (0.5%)28 (1.5%)
29 (1.6%)
45 (2.5%)
8 (0.4%)
21 (1.2%)
0 (0.0%)
13 (0.7%)
1 (0.1%)
7 (0.4%)
1 (0.1%) Adherence ViolationFigure 1: Distribution of annotated social norms across languages, norm categories and status (adherence/violation)
labels. Numbers in parenthesis indicate sample percentage contribution in the database.
terpersonal, and situational cues. Existing datasets
have made significant strides in curating large-scale
corpora for norm recognition and classification.
However, each of these datasets presents notable
limitations as described in Section 2. In contrast,
the proposed MINDS corpus is curated from real
bilingual conversational sessions with rich cultural
and linguistic grounding. Below we outline the
data collection and annotation methodology.
3.1 Data Collection
Each of the 31 sessions features a two-person,
multi-turn dialogue between a foreign language
expert and an English language expert. During
the interaction, the foreign language expert com-
municates solely in the foreign language (either
Mandarin or Spanish), while the English expert
responds entirely in English. This structure was de-
signed to simulate real-world bilingual communica-
tion scenarios such as interpreter training, second-
language learning environments, or multilingual
human-computer interactions.
The dataset is evenly balanced between the two
language groups, where 16 sessions involve Man-
darin speakers and 15 involve Spanish speakers.
The Mandarin subset includes 12 unique speak-
ers, with each unique speaker pair appearing only
once. Similarly, 22 unique Spanish speakers were
recruited with the same one-time unique pair partic-
ipation constraint. This design ensures speaker di-
versity and eliminates redundancy, which is critical
for fair evaluation in speaker-independent model-
ing tasks. The dialogues reflect a wide range of in-
terpersonal dynamics, accents, and conversational
styles. By enforcing the one-language-per-speaker
rule, the setup captures code-switching boundaries,
implicit translation patterns, and culturally specific
communication strategies, making this dataset well-suited for multilingual NLP tasks.
3.2 Annotation Protocol
Each session in the dataset was independently an-
notated by multiple human raters. The annotators
were tasked with labeling individual utterances
within the dialogue, assigning a social norm and
a status (e.g., adherence or violation of the social
norm). These judgments reflect the pragmatic and
sociolinguistic interpretation of the utterances in
context, requiring annotators to consider intent,
tone, and interpersonal dynamics. The annotation
process was structured such that no single annota-
tor was responsible for all sessions in a given lan-
guage group. Instead, annotators were selectively
assigned to sessions, resulting in partial coverage
across sessions and users. For instance, within the
Mandarin subset, 5 unique annotators participated,
with each annotating a subset of the 16 sessions.
Similarly, 6 annotators contributed to the Spanish
subset, with varied levels of session coverage. No
annotator covered all sessions, a deliberate design
choice to maintain annotator diversity and avoid
individual annotator bias dominating the evaluation
outcomes.
Each turn in a session was labeled by at least
one annotator, and many were reviewed by two
or more annotators, enabling inter-rater reliability
(IRR) analysis. In cases where multiple annotators
reviewed the same turn, the agreement was quan-
tified using Cohen’s Kappa across combinations
of norm and status annotations. Several sessions
show high IRR, with only minor annotation dis-
crepancies in labeling across turns, indicating high
consistency where overlap existed. This further
supports the reliability of the annotations in mod-
eling and evaluation tasks. By leveraging a multi-
rater framework and ensuring speaker-annotator

diversity, this annotation protocol supports robust
downstream applications such as norm adherence
classification, multilingual dialogue modeling, and
socially intelligent agent development. The combi-
nation of linguistic variation, dialogue realism, and
reliable annotation makes this dataset a valuable
benchmark for cross-cultural and multilingual AI
systems.
300 298
203
168
88
52
177 1 1200
150
100
50
1 NormNumber of Unique Norms per SampleCount
2 Norms 3 Norms 4 Norms 5 Norms250Mandarin Spanish
Figure 2: Frequency of norm categories per conversa-
tion sample.
3.3 Insights
As illustrated in Figure 1, Mandarin-English dia-
logues show a strong skew toward Persuasion and
Request categories, suggesting a focus on negotia-
tion and directive strategies. In contrast, Spanish-
English interactions are dominated by Request for
Information and Thanks, reflecting a more transac-
tional or expressive discourse style. These trends
highlight language-specific tendencies in conversa-
tional norms. The relative frequency of Disagree-
ment and Criticism in Spanish, compared to Persua-
sion in Mandarin, points to cultural differences in
how interpersonal boundaries and assertiveness are
navigated. Such patterns underscore the need for
culturally adaptive modeling of normative behavior.
Violations are infrequent overall but consistently
concentrated in sensitive categories like Criticism
and Disagreement across both languages. These
acts, though less common, are more likely to devi-
ate from normative expectations, indicating higher
pragmatic risk.
Most utterances reflect norm adherence, suggest-
ing it is the default mode of interaction. However,
norm-sensitive acts such as refusals or criticisms,
even when rare, carry higher likelihoods of vio-
lation highlighting the asymmetry in norm obser-
vance across speech acts. The dataset captures
a broad spectrum of social norms with balanced
coverage of adherence and violation, across two lin-
guistically and culturally distinct bilingual contexts.
This makes it well-suited for cross-cultural norm
modeling and the development of socially aware di-alogue systems. Figure 2 illustrates the distribution
of the number of unique norm categories annotated
per conversation sample, comparing Mandarin-
English and Spanish-English dialogues. Most sam-
ples exhibit a single norm type, with 298 Man-
darin and 203 Spanish samples falling into this
category. As the number of distinct norms per
sample increases, the counts drop sharply, indicat-
ing that multi-norm interactions are less common.
Nevertheless, a notable portion of conversations
particularly in Mandarin contain two or more co-
occurring norms, highlighting the normative com-
plexity present in real-world dialogue. This dis-
tribution supports the need for models capable of
handling multi-label norm classification in conver-
sational contexts.
4 Approach
Here, we describe our Retrieval-Augmented Gener-
ation framework,Norm-RAG, designed for social
norm classification and adherence detection in dia-
logues. Unlike prior works that either retrieve from
static corpora (Forbes et al., 2020) or synthesize
templated dialogue (Li et al., 2023), our method
targets real-time, evolving conversational data. The
central insight ofNorm-RAGis to represent norms
through pragmatic, multidimensional structures, en-
abling semantic and social interpretability beyond
surface-level content. Our framework consists of
four stages as illustrated in Figure 3: (1)Seman-
tic Clustering for Norm Chunking, (2)Structured
Norm Attribute Extraction, (3)Semantic Norm Re-
trieval and Re-ranking, (4)Dialogue-Aware Norm
Classification. Refer to Appendix A for details
about the different prompt templates used through-
out the entire process.
4.1 Semantic Clustering for Norm Chunking
To structure the norm corpus for retrieval and
alignment, we adopt a block-diagonal cluster-
ing approach guided by semantic similarity be-
tween norm sentences. Inspired by prior work
on block-diagonal clustering structures (Xing and
Zhao, 2024), we utilize a similarity-based heuristic
grounded in the observation that semantically co-
herent norm definitions exhibit localized similarity
in embedding space.
Let the norm document consist of nsequential
sentences {x1, x2, ..., x n}. Each sentence xiis
encoded using a pre-trained sentence transformer,
si∈Rd. We define the pairwise cosine similarity

Figure 3: Illustration of the different stages of our proposed framework. Stage I represents semantic clustering
for norm chunking. Stage II & III illustrates the structured norm attribute extraction, semantic norm retrieval and
re-ranking modules. Finally, Stage IV shows the dialogue-aware norm classifier within theNorm-RAGsystem.
matrixS∈Rn×n:
Si,j=si·sj
∥si∥∥sj∥(1)
When visualized as a heatmap, Sreveals a clear
block-diagonal structure, clusters of high similarity
corresponding to contiguous segments of norm-
related sentences. These blocks emerge naturally
because norm definitions and their illustrative ex-
amples tend to form coherent segments within the
text. To segment these into preliminary clusters, we
use a greedy line-wise algorithm. We begin with
sentence x1and append subsequent sentences until
the similarity with the previous line falls below a
segmentation threshold ϵseg. Formally, for each i,
we assignx ito the current cluster if
cos(s i−1, si)≥ϵ seg (2)
Otherwise, a new cluster is initiated. We find that
each cluster closely represents a candidatenorm
chunk Ck, a semantically consistent unit of norm
information. Initial segmentation often leads to
over-fragmentation due to outlier examples or edge-
case scenarios within the same norm description.
To address this, we apply a refinement stage: adja-
cent clusters CkandCk+1are merged if the cosinesimilarity between their mean embeddingsµ kand
µk+1exceeds a merging thresholdϵ merge:
cos(µ k, µk+1)> ϵ merge (3)
Note, in experimentation we used grid search
to find the optimal value for ϵsegandϵmerge. This
two-phase process ensures that each norm chunk
captures a semantically cohesive unit while mini-
mizing cross-norm contamination. In practice, we
observe that a single norm may be split across two
clusters, however, it is rare for a single cluster to
span multiple norms, thereby validating the preci-
sion of our clustering approach. These discovered
norm chunks form the foundational retrieval units
for semantic alignment with dialogue, enabling
fine-grained context-aware norm reasoning in the
subsequent stages of our framework.
4.2 Structured Norm Attribute Extraction
Real-world dialogue often evolves dynamically,
with shifting intents, roles, and relationship fram-
ings. Different from general pipelines, the core
insight driving ourNorm-RAGis that normative
behavior in conversation is multi-dimensional, gov-
erned not just by surface-level semantics. We repre-
sent these multidimensional structures around four
foundational attributes of normative behavior:

1. Communicative Intent (CI):What is the pragmatic
goal conveyed (e.g., apologize, warn, inquire)?
2. Interpersonal Framing (IF):What social relation-
ship or power dynamic is implied (e.g., elder-junior, peer-
peer)?
3. Linguistic Features (LF):Which discourse cues or
syntactic patterns characterize the norm (e.g., hedging,
imperative verbs)?
4. Contextual Triggers and Constraints (CTC):What
environmental, cultural, or situational conditions activate
or constrain the norm?
These dimensions capture latent communicative
goals, social relationships, stylistic strategies, and
situational factors that are essential to identifying
and interpreting normative behavior. Next, we de-
scribe how these attributes are extracted for both
the extracted norm chunks and dialogues during
inference.
Once the norm chunks {Ck}are identified,
we extract their attributes via prompting a LLM,
e.g.,gpt-4o-mini (Hurst et al., 2024). The goal
is to extract the four underlying components
(CI, IF, LF, CTC) from each norm chunk Ckin
a normalized format:
CkPGPT− − − →N Ck:{CI Ck, IFCk, LFCk, CTC Ck}
(4)
The prompt is designed to be culturally sensi-
tive and context-aware, mirroring the structure of
the norm examples provided in the documentation.
The LLM returns a structured JSON-like response
with textual content or canonical tags correspond-
ing to each attribute. This extraction process not
only supports semantic alignment in downstream
retrieval, but also introduces interpretability and
modularity in norm representation. By decoupling
normative knowledge into interpretable attributes,
our method enables precise matching between dia-
logue utterances and applicable social norms, sup-
porting both generalization and cultural specificity
in norm understanding.
4.3 Attribute-based Semantic Retrieval and
Re-ranking
For a dialogue segment Dtup-to time t, our goal
is to retrieve semantically aligned norm definitions
based on shared pragmatic structure. We begin
by extracting a normative attribute vector from Dt
using the same four-part schema introduced earlier:
DtPGPT− − − →N Dt:{CI Dt, IFDt, LFDt, CTC Dt}
(5)
Each attribute ai
Dtis compared independently
against its corresponding attribute ai
Ckfrom allnorm chunks in the index. Let ϕdenote the cosine
similarity between embeddings. We define the ag-
gregated similarity score between NDtandNCk
as:
simk=1
44X
i=1ϕ(ai
Dt, ai
Ck)(6)
We retrieve top- ncandidate norm chunks with
the highest simkvalues, filtered to exceed a global
threshold µsim(average across all simkscores), to
ensure semantic proximity. These candidates serve
as context for grounding downstream prompt gen-
eration.
Dialogue Window Design:To prevent topic
drift from full-dialogue embeddings, we adopt a
recency-focused strategy. Let utbe the utterance
at time tandDtthe entire dialogue segment up till
timet. We construct a focused context window:
¯Dt=Concat(u t−l, . . . , u t)(7)
This structure only considers the current utter-
ance and previous ldialogue turns, thereby helping
preserve critical dependencies while emphasizing
recency. Here, lis determined via LLM given the
dialogue history and the latest utterance. We op-
tionally apply weighted averaging of embeddings
with positional bias toward utto capture moment-
level intent without any prior contextual noise.
Re-ranking Module:The retrieved candidates
are further evaluated using a LLM-based re-ranker.
Given extracted query attributes {ai
Dt}and each
norm chunk Nn, the reranker Revaluates contex-
tual alignment and re-ranks the retrieved norm can-
didates. The top-ranked norm is accompanied by
a natural language explanation generated by the
reranker, which justifies its contextual relevance to
the dialogue. This attribute-disentangled retrieval
mechanism mirrors the interpretive reasoning hu-
mans apply when aligning utterances with social
norms. It also provides a semantically grounded
context that enhances prompt-based inference in
the final classification stage.
4.4 Dialogue-Aware Norm Detection Agent
In the final stage of theNorm-RAGpipeline,
our objective is to identify at each time step t,
the relevant social norm category and status for
each turn within the dialogue session. Addition-
ally, we are also interested in generating expla-
nations for the predicted result and relevant feed-
back to help at the next turn in the dialogue. To

do this our RAG-based agentic framework incre-
mentally processes each dialogue turn, one ut-
terance at a time. At step t, the agent receives
the focused context window ¯Dt; re-ranked re-
trieved norm chunks{N∗
1, . . . , N∗
n}; and dialogue
attributes NDt:{CI Dt, IFDt, LFDt, CTC Dt}.
These along with the dynamic feedback variable
Ft−1from the previous turn is embedded as a struc-
tured prompt and presented to the LLM for clas-
sification. Note, the feedback encodes high-level
observations (e.g., tone, contradiction, escalation)
that can influence future norm adherence. For the
first utterance (t= 1), we setF 0=∅.
LLM Query and Output:The different LLM
query inputs and outputs are illustrated as follows:
LLM Inference at Each Step
At each utterance step t, the LLM is prompted
with:
• Focused Dialogue History: ¯Dt
• Retrieved Chunk Attributes:CI Dt, IFDt,
LFDt, CTC Dt
• Retrieved Norm Context:{N∗
1, . . . , N∗
n}
• Prior Feedback:F t−1
The model produces:
• Norm Category:y t
• Norm Status:s t∈ {adhered,violated}
• Explanation fory tands t
• Updated Feedback Signal:F t
Formally, the model performs:
(yt, st, Ft) =L( ¯Dt, CI t, IF t, LF t, CTC t,{N∗
i}k
i=1, Ft−1)
(8)
where Ldenotes the LLM invoked with a cus-
tom instruction-tuned prompt. This agentic setup
enables feedback-driven norm reasoning over the
course of an entire conversation.
Agentic Loop:This structure forms a dynamic
reasoning loop over the session:Agentic Reasoning Loop
For a given session with utterances
{u1, . . . , u T}:
1. InitializeF 0=∅
2. Fort= 1toT:
• Extract ¯Dt={u 1, ..., u t−1}
• Retrieved Chunk Attributes:CI Dt, IFDt,
LFDt, CTC Dt
• Prompt Construction: ¯Dt,NDt, andF t−1
• Query LLM to obtain(y t, st, Ft)
3. Repeat until end of session
By iteratively grounding each utterance in struc-
tured context, retrieved norms, and conversational
dynamics, theNorm-RAGagent enables robust
and explainable norm classification throughout
multi-turn interactions. This agentic structure also
supports temporal coherence, capturing how nor-
mative behavior evolves within a session.
5 Experiments
We conduct a comprehensive evaluation of our
proposed approach on norm discovery and adher-
ence/violation classification for conversations that
include multi-lingual and cross-culture scenarios.
There has been no baselines or proposed model
for Norm Discovery and Adherence discovery ex-
cluding NormSage (Fung et al., 2022) and Norm-
Dial (Li et al., 2023). However, these works do
not compare to our proposed dataset and metric.
Hence, we include the latest state-of-the-art LLMs
both proprietary (GPT-4o (Hurst et al., 2024))
and open-sourced (LLaMA (Dubey et al., 2024),
QWEN (Yang et al., 2025), Phi (Abdin et al., 2024))
baselines for comparison. For each utterance
5.1 Norm Classification and Adherence
Detection Accuracy
Given a conversation, the task is to classify which
social norm categories are invoked by thelatest
utterance, and determine whether the utterance ad-
heres to or violates them. We evaluate this across
two subtasks: norm classification and adherence
status detection.
We benchmark ourNorm-RAGframework
against a range of zero-shot LLM baselines both
closed-source (GPT-4o-mini) and open-source
(LLaMA 3.1 8B, Qwen-3 32B, Phi-4 14B), across
two key input configurations: 1.Hist.: Whether
the model is provided with only thelatest utterance

Model Hist. Docs Norm
Acc.Adh./Viol.
Acc.
GPT-4o-mini✗ ✗64.6 57.2
Ours w/ GPT-4o✓ ✓70.4 63.6
LLaMA 3.1 (8B)✗ ✗56.4 44.7
LLaMA 3.1 (8B)✓ ✗57.4 45.4
Ours w/ LLaMA✓ ✓64.4 54.4
Qwen-3 (32B)✗ ✗61.1 54.9
Qwen-3 (32B)✓ ✗62.3 57.1
Ours w/ Qwen✓ ✓67.9 60.2
Phi-4 (14B)✗ ✗66.4 58.6
Phi-4 (14B)✓ ✗67.861.6
Ours w/ Phi✓ ✓69.160.2
Table 1:Accuracy for norm classification and adherence/vio-
lation detection on the MINDS corpus.Hist.: dialogue history
used (✓) vs. only last utterance ( ✗).Docs: retrieved documen-
tation used (✓) or not (✗).
or with theentire conversation history + latest ut-
terance, 2.Docs.: Whether the model receivesno
external contextor is givenretrieved documenta-
tioncontaining culture-specific norm definitions
and examples. Each baseline is evaluated under
these combinations, while ourNorm-RAGmethod
additionally integrates context i.e., structured at-
tributes and feedback-driven reasoning.
As shown in Table 1 and Table 2, our approach
consistently outperforms all baseline configura-
tions across both norm classification and adherence
detection tasks. Within each model family, the
addition of historical context and external norma-
tive knowledge leads to meaningful gains.Norm-
RAGfurther amplifies this by grounding LLM in-
ference in retrieved, semantically-aligned norm def-
initions.Norm-RAGachieves an improvement of
+14.1% ,+21.7 improvement for LLaMA 3.1 in
norm classification and adherence detection accu-
racy respectively and lowest for Phi-4 with +4.%
in norm classification while a drop in −2% in
adherence accuracy. The modest gains with Phi-
4 suggest that its performance benefits less from
additional context, likely due to its reliance on in-
ternal knowledge and preference for concise in-
structions—larger retrieved prompts may disrupt
its reasoning compared to models like Qwen and
LLaMA that better utilize external context. Over-
all our results confirms that retrieval-augmented
structure-aware prompting yields stronger general-
ization across dialog settings.
5.2 Ablation Study
To better understand the contribution of differ-
ent components in theNorm-RAGframework,Model Hist. Docs Norm
Acc.
GPT-4o-mini✓ ✗58.0
Ours w/ GPT-4o✓ ✓62.0
LLaMA 3.1 (8B)✓ ✗57.2
Ours w/ LLaMA✓ ✓62.0
Qwen-3 (32B)✓ ✗52.4
Ours w/ Qwen✓ ✓55.0
Table 2:Accuracy for norm classification and adherence/vi-
olation detection on the SocialDial corpus.Hist.: dialogue
history used ( ✓) vs. only last utterance ( ✗).Docs: retrieved
documentation used (✓) or not (✗).
Ablation Variant Qwen-3 GPT-4o
Fixed text split63.2/58.7 66.1/61.2
Cluster w/o attr.65.1/59.0 67.4/61.9
w/o feedback67.0/60.1 69.2/62.9
Full pipeline67.9/60.2 70.4/63.6
Table 3:Ablation study of theNorm-RAGpipeline. "Fixed
text split" uses uniform segmentation; "Cluster w/o attr." ex-
cludes attribute extraction; "w/o feedback" disables conver-
sational feedback. We report norm classification and status
accuracy as Norm/Status.
we conduct an ablation study summarized in Ta-
ble 3. Starting with a fixed-size text splitter chunk-
ing mechanism (row 1), we observe significantly
lower performance across both models, suggest-
ing that arbitrary chunking limits semantic cohe-
sion in the retrieved context. When switching
to semantically clustered chunks without extract-
ing norm attributes (row 2), we see consistent im-
provements (+1.9 in Qwen and +1.3 in GPT for
norm accuracy), indicating the value of topically
coherent segmentation. Turning off feedback (row
3) yields additional gains, showing that conversa-
tional grounding plays a role in cumulative norm
reasoning. Finally, the fullNorm-RAGpipeline
(row 4) achieves the best performance across both
models, confirming that attribute-guided context
and feedback-driven prompting are complementary.
Notably, GPT shows a larger gain from feedback
than Qwen, likely due to its stronger adaptation to
turn-level dialog intent modeling.
5.3 Qualitative Analysis
In challenging cases involving overlapping intents,
such as: “I want the junior surgeon who performed
the operation to be fired."Norm-RAGcorrectly

identifies the doing request as an act of adher-
ence, while also detecting the embedded doing
criticism as a norm violation—matching the
human-labeled ground truth. By leveraging struc-
tured dialog attributes and semantically retrieved
normative context, Norm-RAG disentangles multi-
intent utterances and reasons about their distinct
normative implications with high fidelity, even
when cues are emotionally charged or culturally
sensitive.
6 Conclusion
We presentedNorm-RAG, a retrieval-augmented,
agentic framework for modeling social norm ad-
herence in multilingual dialogue. By combining
semantically clustered norm documentation with
structured dialog attributes—such as communica-
tive intent, framing, and linguistic features—our
approach enables interpretable and culturally sen-
sitive norm reasoning. We introducedMINDS, a
novel bilingual dataset of Mandarin-English and
Spanish-English conversations annotated for norm
categories and adherence status. Empirical re-
sults across multiple LLMs demonstrate significant
gains in both norm classification and adherence
detection. Through ablation studies, we show the
importance of semantic chunking, structured at-
tribute modeling, and feedback-based prompting.
Together, these components move beyond static,
template-based inference toward dynamic, socially
grounded interaction. Future work will explore
higher-order norm dynamics such as escalation,
social repair, and longitudinal alignment in multi-
lingual, multi-agent settings.
Acknowledgments
This material is based upon work supported by
the Defense Advanced Research Projects Agency
(DARPA) under Contract No. HR001122C0032.
Any opinions, findings and conclusions or recom-
mendations expressed in this material are those
of the author(s) and do not necessarily reflect the
views or policies of DARPA, the Department of
Defense or the U.S. Government. Additionally,
we acknowledge the use of GPT-based language
models during the development of this work for
research assistance and code prototyping. These
tools supported various stages of exploration, anal-
ysis, and implementation.Limitations and Ethical Considerations
While our work aims to improve norm understand-
ing in multilingual dialogues, it has several lim-
itations. First, our dataset covers only two non-
English languages (Mandarin and Spanish), which
limits generalization to other linguistic and cultural
contexts. Second, although we employ LLMs for
attribute extraction and classification, these mod-
els can inherit social and cultural biases from their
training data, which may affect norm interpretation.
We also acknowledge the limitations of using sim-
ulated or partially controlled bilingual dialogues,
which may differ from spontaneous in-the-wild in-
teractions.
Ethically, we ensured speaker anonymization
in our dataset and limited language use to non-
sensitive topics. We encourage future work to ex-
pand linguistic coverage and explore real-world
deployment risks in downstream applications.
Computational ResourcesWe conducted all ex-
periments using 2× NVIDIA A6000 GPUs (each
with 48GB VRAM). The classification pipeline
used three open-source LLMs: LLaMA 3.1 (8B),
Qwen (32B), and Phi-3 (14B), applied to a dataset
of 835 utterances. For each utterance, the agent
performed structured prompting and reranking per
model. The total compute time across all models
was approximately 30 GPU hours on the A6000
infrastructure.
References
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien
Bubeck, Ronen Eldan, Suriya Gunasekar, Michael
Harrison, Russell J Hewett, Mojan Javaheripi, Piero
Kauffmann, and 1 others. 2024. Phi-4 technical re-
port.arXiv preprint arXiv:2412.08905.
Yehuda Arieli. 1964. Cultural factors in communication
effectiveness.International Journal of Intercultural
Relations, 1(2):143–155.
Sky CH-Wang, Arkadiy Saakyan, Oliver Li, Zhou Yu,
and Smaranda Muresan. 2023. Sociocultural norm
similarities and differences via situational alignment
and explainable textual entailment.arXiv preprint
arXiv:2305.14492.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
Denis Emelin, Ronan Le Bras, Jena D Hwang, Maxwell
Forbes, and Yejin Choi. 2020. Moral stories: Situated

reasoning about norms, intents, actions, and their
consequences.arXiv preprint arXiv:2012.15738.
Maxwell Forbes, Jena D. Hwang, Vered Shwartz,
Maarten Sap, and Yejin Choi. 2020. Social chem-
istry 101: Learning to reason about social and moral
norms. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 653–670, Online. Association for
Computational Linguistics.
Yi R Fung, Tuhin Chakraborty, Hao Guo, Owen
Rambow, Smaranda Muresan, and Heng Ji. 2022.
Normsage: Multi-lingual multi-cultural norm discov-
ery from conversations on-the-fly.arXiv preprint
arXiv:2210.08604.
Yuling Gu, Bhavana Dalvi Mishra, and Peter Clark.
2021. Dream: Improving situational qa by
first elaborating the situation.arXiv preprint
arXiv:2112.08656.
Jonathan Haidt. 2012.The righteous mind: Why good
people are divided by politics and religion. Vintage.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Liwei Jiang, Jena D Hwang, Chandra Bhagavatula, Ro-
nan Le Bras, Jenny Liang, Jesse Dodge, Keisuke
Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia
Gabriel, and 1 others. 2021. Can machines learn
morality? the delphi experiment.arXiv preprint
arXiv:2110.07574.
Hyunwoo Kim, Youngjae Yu, Liwei Jiang, Ximing
Lu, Daniel Khashabi, Gunhee Kim, Yejin Choi, and
Maarten Sap. 2022. Prosocialdialog: A prosocial
backbone for conversational agents.arXiv preprint
arXiv:2205.12688.
Oliver Li, Mallika Subramanian, Arkadiy Saakyan, Sky
CH-Wang, and Smaranda Muresan. 2023. Normdial:
A comparable bilingual synthetic dialog dataset for
modeling social norm adherence and violation.arXiv
preprint arXiv:2310.14563.
Hannah Rashkin, Maarten Sap, Emily Allaway, Noah A
Smith, and Yejin Choi. 2018. Event2mind: Com-
monsense inference on events, intents, and reactions.
arXiv preprint arXiv:1805.06939.
Maarten Sap, Ronan Le Bras, Emily Allaway, Chan-
dra Bhagavatula, Nicholas Lourie, Hannah Rashkin,
Brendan Roof, Noah A Smith, and Yejin Choi. 2019.
Atomic: An atlas of machine commonsense for if-
then reasoning. InProceedings of the AAAI con-
ference on artificial intelligence, volume 33, pages
3027–3035.
Shalom H Schwartz, Jan Cieciuch, Michele Vecchione,
Eldad Davidov, Ronald Fischer, Constanze Beierlein,
Alice Ramos, Markku Verkasalo, Jan-Erik Lönnqvist,Kursad Demirutku, and 1 others. 2012. Refining
the theory of basic individual values.Journal of
personality and social psychology, 103(4):663.
Muzafer Sherif. 1936. The psychology of social norms.
Harry C. Triandis. 1994.Culture and Social Behavior.
McGraw-Hill, New York.
Zheng Xing and Weibing Zhao. 2024. Block-diagonal
guided dbscan clustering.IEEE Transactions on
Knowledge and Data Engineering, 36(11):5709–
5722.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Haolan Zhan, Zhuang Li, Xiaoxi Kang, Tao Feng,
Yuncheng Hua, Lizhen Qu, Yi Ying, Mei Rianto
Chandra, Kelly Rosalin, Jureynolds Jureynolds, and 1
others. 2024. Renovi: A benchmark towards remedi-
ating norm violations in socio-cultural conversations.
InFindings of the Association for Computational
Linguistics: NAACL 2024, pages 3104–3117.
Haolan Zhan, Zhuang Li, Yufei Wang, Linhao Luo,
Tao Feng, Xiaoxi Kang, Yuncheng Hua, Lizhen Qu,
Lay-Ki Soon, Suraj Sharma, and 1 others. 2023. So-
cialdial: A benchmark for socially-aware dialogue
systems. InProceedings of the 46th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 2712–2722.
Caleb Ziems, Jane Dwivedi-Yu, Yi-Chia Wang, Alon
Halevy, and Diyi Yang. 2023. Normbank: A knowl-
edge bank of situational social norms.arXiv preprint
arXiv:2305.17008.
Caleb Ziems, Jane A Yu, Yi-Chia Wang, Alon Halevy,
and Diyi Yang. 2022. The moral integrity corpus:
A benchmark for ethical dialogue systems.arXiv
preprint arXiv:2204.03021.

A Prompts Used in Norm-RAG
Norm Attribute Extraction Prompt
System Prompt
""" You are a social interaction analyst specializing in pragmatics and social
norm recognition in conversation .
Given the dialogue below , extract a structured representation of the
** speaker's behavior in the final utterance **, focusing on how it performs or
aligns with one or more socially recognizable norms such as persuasion ,
request , refusal , apology , etc .
The extracted structure will be used to retrieve similar conversational
behaviors , so it must ** accurately reflect the speech act's social function ,
nuance , and framing **, in a way that can ** disambiguate between multiple norm
categories **.
=> If the utterance aligns with ** more than one norm ** (e.g.,'doing request'
+'doing thanks'), your attributes should reflect that layered action .
Return the following 4 ** pragmatic attributes **:
```json
{{
``CommunicativeIntent ":``<Describe *allcommunicative goals * the speaker
ispursuing -- both primaryandsecondary . Use norm languageifapplicable
(e.g., persuading , requesting info , refusing , finalizing ). Prioritize
intent differentiation across norms .>",
``InterpersonalFraming ":``<How the speaker * relates to the listener *:
formality , power dynamics , face - work ( saving / threatening ), emotional
stance ,oralignment . Make distinctions like deferential vs. assertive ,
affiliative vs. distancing --asthey cue norm categories .>",
``LinguisticFeatures ":``<Detail rhetorical strategies used to * signalor
mitigate norm performance *: hedges , indirectness , modality (e.g.,'might',
'should'), discourse markers , politeness formulas , etc . Capture evidence
that helps distinguish one normfromanother .>",
``ContextualTriggersAndConstraints ":``<What about the broader dialogueor
situation shapes how this normisperformed ? Include role relations ,
timing , known stakes , prior acts , social rulesorexpectations that
constrain the speaker's behavior .>"
}}
```
"""
User Prompt
### Dialogue Context :
{ dialog_context }

Dialog Window Design Prompt
System Prompt
You are an expertinpragmaticsandsocial norms .
Given the dialogue history below , analyze the communicative functionand
social dynamic of the most recent turn .
Pleasereturnyour responseinthe following JSONformat:
```
{{
" CommunicativeIntent ": "<short summary of what the speaker is trying to
achieve >",
" InterpersonalTension ": "<comment on any social tension , repair ,
dominance , submission , etc .>",
" LikelyNormCategory ": "<the most likely norm involved , e.g.,'doing
apology','doing greeting', etc .>",
" ContextDependenceScore ": <floatbetween 0.0and1.0 , where higher means
more dependent on prior context >
}}
```
User Prompt
### Dialogue History :
{ dialog_history }

Re-ranking Retrieved Norm Chunks Prompt
Prompt
You are a pragmaticsanddiscourse analysis expert .
You are given :
-- A brief snippet of dialogue ( usually the last 1-2 turns of a conversation ),
-- A structured interpretation of that snippet ,forattribute
{ attribute_name },
-- Alistof candidate norm definitions retrievedfroma semantic search
system .
Your taskisto rerank the candidatesfrommost to least relevant , based on
how well each one alignswiththe communicative behavior expressedinthe
dialogueasrepresented by the extracted attributes .
### Dialogue Context :
"{ dialog_context }"
### Extracted Norm Attributes :
{ attributes }
### Retrieved Candidate Norm Descriptions :
{ doc_entries }
### Instructions :
-- Compare the overall meaningandfunction of each candidate to the
extracted attributes .
-- Pay special attention to the Communicative Intent , but also consider
whether the interpersonal stance , language choices ,andsituational framing
match .
-- Your goalisto rank which candidate best captures thetypeof norm being
enactedinthe given dialogue .
### Output Format :
{{
" Ranking ": [3, 1, 2, 4, 5],
" TopJustification ": " ... "
}}
Onlyreturnthe JSONobject.

Feedback Prompt
System Prompt
You are a pragmatic analyst helping to generate interpretive contextfor
understanding turn -by - turn normsinconversation .
Given the most recent utteranceina dialogue , alongwithits predicted
norm (s)andsurrounding dialogue context , your taskisto produce ** feedback
that captures the communicative forceandsocial trajectory ** of the current
moment .
This feedback will be used to inform the interpretation of the *next*
utterance -- by helping identify what normsorresponses are socially
relevantorexpected ,andwhat social constraints are alreadyinplay .
### INPUT :
-`DialogueHistory`: The full dialogue history leading up to the latest
utterance ( shortor long).
-`LastUtterance`: The final utterance by the most recent speaker .
-`PredictedNorms`: Oneormore social norms inferredfromthe last
utterance . Oneormore of:
['Doing persuasion','Doing request','Doing requesting information',
'Doing criticism','Doing thanks','Doing greeting','Doing admiration',
'Doing disagreement','Doing refusing a request','Doing apology','Doing
taking leave','Doing granting a request','Doing finalizing
negotiation / deal','No Norm']
### OUTPUT FORMAT :
{
``SituatedSummary ":``<Explain what is being socially performed in the last
utterance , and how it connects to the unfolding dialogue -- including tone ,
intentions , relational shifts , or embedded expectations .>",
``NormImplications ":``<What social norm (s) are being enacted or invoked ?
Why ? Include cues from wording , context , or sequencing .>",
``NextTurnExpectation ":``<What types of responses -- in terms of social
action or stance -- are made relevant by this utterance ? What does it
* invite *, * pressure *, or * allow * the next speaker to do (or not do)?
Mention if there's a power dynamic , politeness constraint , emotional
charge , etc .>"
}
User Prompt
### Dialogue History
{ dialoghistory }
### Last Utterance
{ lastutterance }
### Predicted Norms
{ predictednorms }

Norm Detection and Adherence Detection Prompt
System Prompt
You are an expertinanalyzing conversations to identify underlying social
norms . Your taskisto classifyallapplicable social norm categories
( minimum 2, maximum upto 5) reflectedinthe ** latest utterance ** of a given
dialogue using both ** explicitandimplicit cues ** of social interaction .
### Norm Categories :
{ norm_categories }
### Task Instructions :
1. Use the ** entire dialogue history **andthe ** retrieved contextfromRAG **
to interpret the ** social intent ** behind the ** latest utterance **.
- Consider both ** explicit speech acts ** (e.g., asking , refusing )and
** implicitorindirect signals ** (e.g., persuading by justification ,
criticizing through description ).
- Understand the progressionandstructure of the dialogue to reveal the
** pragmatic function ** of the utterance .
2. Identify **allrelevant norm categories ** the latest utterance satisfies
fromthelist( maximum 3).
- Choose norms based on ** intent **, ** emotion **, ** relational context **,
** dialogue progression **,and** linguistic cues **, even when ** indirectly
expressed **.
- Include ** weakormoderate instances ** of norms (e.g., subtle persuasion
orsoft disagreement ),notjust overt ones .
3. For each norm category :
- Assess whether the utterance reflects ** Adherence **or** Violation ** of
that norm .
4. Evaluate whether the ** retriever context **isrelevant to the ** overall
setof predicted norms **:
- If ** Relevant **, use it to support a more confident classification .
- If ** Not Relevant **, ignore the retriever contextanduse your own
reasoning about social norms .
5. If ** no identifiable norm **ispresentinthe utterance :
- Return only one entry :
- Norm Category :`No Norm`
- Status :`Violation`
6. Provide a ** natural language confidence level **foryour prediction :
- Choosefrom:`High`,`Medium`,or`Low`
- Justify your confidence based on clarity of social intent , surfaceand
hidden patterns ,andcontext fit .
### Output Format in JSON :
```json
{{
" latest_utterance ": "<copy of the utterance >",
" predicted_norms ": [
{{" norm_category ": "<norm category 1>", " status ": "<Adherence or
Violation >"}},
{{" norm_category ": "<norm category 2>", " status ": "<Adherence or
Violation >"}},
{{" norm_category ": "<norm category 3>", " status ": "<Adherence or
Violation >"}}
],
" retriever_context_relevance ": "<Relevant / Not Relevant >",
" confidence_level ": "<High / Medium / Low >",
" explanation ": "<Justify the norm predictions , referencing how context and
implicit cues shaped the interpretation >"
}}
```
User Prompt with (or without) Context
{%ifcontext %}
### Relevant Context from RAG on 4 key attributes that are used to capture
the underlying norm :
{{ context }}
{% endif %}
### Dialog :
{{ dialog }}