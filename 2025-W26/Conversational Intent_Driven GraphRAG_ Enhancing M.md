# Conversational Intent-Driven GraphRAG: Enhancing Multi-Turn Dialogue Systems through Adaptive Dual-Retrieval of Flow Patterns and Context Semantics

**Authors**: Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang, HanGeng Chen, Mengran Zhang, Xilun Chen

**Published**: 2025-06-24 07:20:45

**PDF URL**: [http://arxiv.org/pdf/2506.19385v1](http://arxiv.org/pdf/2506.19385v1)

## Abstract
We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval
Augmented Generation), a novel framework that addresses the limitations of
existing dialogue systems in maintaining both contextual coherence and
goal-oriented progression in multi-turn customer service conversations. Unlike
traditional RAG systems that rely solely on semantic similarity (Conversation
RAG) or standard knowledge graphs (GraphRAG), CID-GraphRAG constructs dynamic
intent transition graphs from goal achieved historical dialogues and implements
a dual-retrieval mechanism that adaptively balances intent-based graph
traversal with semantic search. This approach enables the system to
simultaneously leverage both conversional intent flow patterns and contextual
semantics, significantly improving retrieval quality and response quality. In
extensive experiments on real-world customer service dialogues, we employ both
automatic metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG
significantly outperforms both semantic-based Conversation RAG and intent-based
GraphRAG baselines across all evaluation criteria. Quantitatively, CID-GraphRAG
demonstrates substantial improvements over Conversation RAG across automatic
metrics, with relative gains of 11% in BLEU, 5% in ROUGE-L, 6% in METEOR, and
most notably, a 58% improvement in response quality according to LLM-as-judge
evaluations. These results demonstrate that the integration of intent
transition structures with semantic retrieval creates a synergistic effect that
neither approach achieves independently, establishing CID-GraphRAG as an
effective framework for addressing the challenges of maintaining contextual
coherence and goal-oriented progression in knowledge-intensive multi-turn
dialogues.

## Full Text


<!-- PDF content starts -->

arXiv:2506.19385v1  [cs.AI]  24 Jun 2025Conversational Intent-Driven GraphRAG:
Enhancing Multi-Turn Dialogue Systems through
Adaptive Dual-Retrieval of Flow Patterns and Context Semantics
Ziqi Zhu, Tao Hu, Honglong Zhang, Dan Yang,
HanGeng Chen, Mengran Zhang, Xilun Chen

Abstract
We present CID-GraphRAG (Conversational Intent-Driven Graph Retrieval Augmented Generation), a
novel framework that addresses the limitations of existing dialogue systems in maintaining both contextual
coherence and goal-oriented progression in multi-turn customer service conversations. Unlike traditional
RAG systems that rely solely on semantic similarity (Conversation RAG) or standard knowledge graphs
(GraphRAG), CID-GraphRAG constructs dynamic intent transition graphs from goal achieved historical
dialogues and implements a dual-retrieval mechanism that adaptively balances intent-based graph traversal
with semantic search. This approach enables the system to simultaneously leverage both conversional
intent flow patterns and contextual semantics, significantly improving retrieval quality and response
quality. In extensive experiments on real-world customer service dialogues, we employ both automatic
metrics and LLM-as-judge assessments, demonstrating that CID-GraphRAG significantly outperforms
both semantic-based Conversation RAG and intent-based GraphRAG baselines across all evaluation
criteria. Quantitatively, CID-GraphRAG demonstrates substantial improvements over Conversation RAG
across automatic metrics, with relative gains of 11% in BLEU, 5%in ROUGE-L, 6%in METEOR, and
most notably, a 58% improvement in response quality according to LLM-as-judge evaluations. These
results demonstrate that the integration of intent transition structures with semantic retrieval creates
a synergistic effect that neither approach achieves independently, establishing CID-GraphRAG as an
effective framework for addressing the challenges of maintaining contextual coherence and goal-oriented
progression in knowledge-intensive multi-turn dialogues.
1 Introduction
Large Language Models (LLMs) excel in text
understanding and generation but struggle with
domain-specific multi-turn dialogues, often pro-
ducing inconsistent or contextually inappropriate
responses (Lewis et al., 2020). While Retrieval-
Augmented Generation (RAG) frameworks ad-
dress these issues using external knowledge, con-
ventional RAG approaches face unresolved chal-
lenges in specialized domains like customer ser-
vice.
Multi-turn customer service dialogues present
two key challenges unaddressed by existing RAG
methods. First, temporal dependencies creates
complex reference chains requiring resolution
across multiple turns (e.g., ”How long will that
take?” needing prior context). Second, conversa-
tional objective evolves dynamically across con-
versation phases (e.g., information-gathering →
diagnosis →solution implementation). These chal-
lenges demand a retrieval approach that captures
both semantic context and conversation structure.
Current research has pursued two separate
approaches to address these challenges: (1)
Conversation RAG methodologies[(ResearchGate,
2025), (Ye et al., 2024)] focus on optimizing
dialogue history representation to enhance re-
trieval precision through semantic matching; (2)
GraphRAG approaches[(Zhu et al., 2025),(Edge
et al., 2024),(Zhang et al., 2025a)] leverage knowl-
edge graphs to model static domain knowledge
and facilitate complex reasoning. However, thesemethodologies have developed in parallel trajec-
tories with fundamental limitations: Conversa-
tion RAG captures semantic similarity but lacks
structural guidance for conversation flow, while
GraphRAG models domain facts but not the dy-
namic intent transitions essential for coherent
multi-turn dialogues.
To address these limitations, we propose CID-
GraphRAG (Conversational Intent-Driven Graph
Retrieval Augmented Generation), a novel frame-
work that integrates intent transition graphs with se-
mantic similarity retrieval. Our approach leverages
goal achieved historical conversations to model
conversation dynamics through both semantic con-
tent and conversation flow, enabling more coherent
and goal-oriented responses. The key contributions
of our work include:
•Intent Transition Graph Construction: We intro-
duce a method to automatically build dynamic
intent transition graphs from goal achieved his-
torical dialogues, systematically cataloging hi-
erarchical intents from both system and user
utterances to create a network of effective con-
versational trajectories.
•Dual-Pathway Adaptive Retrieval: We propose
a novel retrieval approach that balances intent
transition graph traversal with semantic similar-
ity search, through an adaptive weighting mech-
anism. This integration enables the system to
simultaneously leverage conversation structure
and contextual semantics, addressing the limi-
tations of single-signal approaches in complex

multi-turn dialogues.
Experiments on real-world customer service dia-
logues demonstrate that CID-GraphRAG signif-
icantly outperforms both semantic-only retrieval
and intent-only graph approaches across all evalua-
tion metrics, highlighting the effectiveness of our
integrated dual-pathway approach.
2 Related Works
2.1 Retrieval-Augmented Generation (RAG)
RAG enhances Large Language Models (LLMs)
by integrating external knowledge sources, im-
proving response accuracy and informativeness in
tasks like question answering and dialogue sys-
tems (Lewis et al., 2020). Recent advancements
include DH-RAG, which uses dynamic historical
context to simulate human-like reasoning through
long/short-term memory integration (Zhang et al.,
2025b), and SELF-multi-RAG, enabling LLMs to
autonomously determine retrieval timing and query
optimization based on conversation context (Asai
et al., 2023). The MTRAG benchmark provides
specialized evaluation metrics for multi-turn dia-
logue RAG systems (Katsis et al., 2025). Innova-
tive applications extend RAG’s capabilities: Xu et
al. combined knowledge graphs with RAG to main-
tain structural relationships in customer service
data (Xu et al., 2024), while Veturi et al. developed
a decision support system integrating real-time
problem detection with FAQ retrieval (Agrawal
et al., 2024).
2.2 GraphRAG
GraphRAG enhances traditional RAG by integrat-
ing knowledge graphs to enable structured se-
mantic retrieval and multi-hop reasoning, using
entity-relationship representations to improve com-
plex query handling(Zhu et al., 2025).Compared
to text-based RAG, it better captures contextual
dependencies in multi-turn dialogues and resolves
challenges like cross-document information link-
ing(Edge et al., 2024).
Key advantages in customer service include: En-
hanced retrieval accuracy through preserved struc-
tural relationships in knowledge graphs (Xu et al.,
2024); Effective processing of multi-hop queries
requiring cross-fragment reasoning [(Zhu et al.,
2025),(Zhang et al., 2025a)]; Context continuity
maintenance across dialogue turns via entity rela-
tionship tracking (Xu et al., 2024), complementingDH-RAG’s dynamic context management (Zhang
et al., 2025b).
The framework mitigates text segmentation is-
sues in traditional RAG while improving answer
interpretability through visualized reasoning paths
(Linders and Tomczak, 2025), particularly bene-
fiting complex service scenarios with interlinked
conversational dependencies.
2.3 Conversation RAG
Conversation RAG addresses multi-turn dialogue
challenges in customer service, where user queries
often depend on historical interactions, requiring
systems to manage evolving contexts and avoid
redundant retrievals (ResearchGate, 2025). These
techniques enhance contextual relevance and re-
sponse coherence across dialogue turns (Ye et al.,
2024). Key advancements include:
•Context-aware frameworks: DH-RAG (Zhang
et al., 2025b) and SELF-multi-RAG (Asai et al.,
2023) optimize historical context modeling;
• Structured retrieval: KG-RAG (Xu et al., 2024)
leverages knowledge graphs for improved accu-
racy;
•Query decomposition: Collab-RA (Xu et al.,
2025) breaks down complex questions via col-
laborative training;
•Performance benchmarks: RAD-Bench (Kuo
et al., 2024) and MTRAG (Katsis et al., 2025)
enable systematic evaluation. ChatQA (Liu
et al., 2024) demonstrates open-source models
can achieve state-of-the-art conversational QA
through retriever fine-tuning. While standard
RAG remains foundational, GraphRAG excels
in structured reasoning and Conversation RAG
advances context-aware interactions, yet no uni-
versal solution exists.
3 Methodology
CID-GraphRAG innovatively integrates Conversa-
tion RAG and GraphRAG to address the challenge
in current multi-turn customer service dialogue
systems of simultaneously maintaining contextual
coherence and goal-oriented progression. Unlike
traditional RAG approaches, CID-GraphRAG of-
fers the following key advantages:
1.IntenGraph: Differing from conventional graphs,
we propose a novel dual-layer intent graph. This

replaces primitive nodes with dual-layer intent
nodes, dynamically extracting expert intent tran-
sition processes from high-quality historical as-
sistant dialogues to generate a domain-specific
expert intent transition map.
2.Dual-layer Retrieval: Moving beyond traditional
RAG systems that rely solely on semantic simi-
larity (Conversation RAG) or static knowledge
graphs (GraphRAG), we innovatively integrate
the intent transition map with semantic simi-
larity retrieval. This dual-layer retrieval mech-
anism enables mutually-assisted, adaptive re-
trieval, enhancing both retrieval effectiveness
and efficiency.
3.Response Generation Fusing Latent Intent and
Key Knowledge: Leveraging the high-quality
knowledge generated by CID-GraphRAG as
Few-shot examples, the system fuses assistant
and user intent with memory to generate re-
sponses. These responses are more contextu-
ally appropriate and align closely with the cor-
responding reply intent and response strategies
matching the user’s underlying goals.
3.1 System Overview
CID-GraphRAG operates in two distinct phases:
the Construction Phase and the Inference Phase,
with the key components of each phase illustrated
in the accompanying figure (as shown in Figure 1).
•Construction Phase: Leverages an LLM (Large
Language Model) to perform intent generation
on high-quality goal achieved dialogues, subse-
quently constructing the intent graph.
•Inference Phase: Focuses on three core compo-
nents:
1.Intent Identification: Employs an LLM to
identify the user’s intent, utilizing the same
methodology established in the Construction
Phase.
2.Dual-layer Retrieval: Moving beyond tradi-
tional RAG systems that rely solely on seman-
tic similarity (Conversation RAG) or static
knowledge graphs (GraphRAG), we inno-
vatively integrate the intent transition map
with semantic similarity retrieval. This dual-
layer retrieval mechanism enables mutually-
assisted, adaptive retrieval, enhancing both
retrieval effectiveness and efficiency.3.Response Generation: Utilizes the results of
the dual-layer retrieval as few-shot examples.
An LLM then generates the final response by
integrating the identified user intent, assistant
intent, and the current dialogue memory.
Figure 1: CID-GraphRAG Framework: Dual-Layer
Intent-Aware Retrieval Architecture,The architecture il-
lustrates both preprocessing (Intent Graph Construction
& Intent Recognition) and inference phases. During
inference, the system utilizes dual-pathway retrieval
(Intent Retrieval via CID-Graph & History Conversa-
tion Retrieval) combined with a Large Language Model
(LLM) for contextual response generation.
3.2 Construction Phase
During the graph construction phase, the primary
focus is utilizing an LLM to perform intent iden-
tification on high-quality goal achieved historical
dialogues. Based on the actual intent transitions
observed, the intent graph is then constructed (as
depicted in Figure 2).
Figure 2: Internal structure of the CID-Graph.The graph
comprises distinct user intent and assistant intent nodes
(categorized as Level 1 or Level 2) and history conver-
sation nodes. Key relations include hierarchical (has
parent/child intent), pairing (in pair with), dialogue an-
choring (has conversation), and sequential flow (next
intent).
3.2.1 Intent Recognition
Traditional GraphRAG constructs knowledge
graphs using single-level intent representations.
This approach often leads to semantic conflicts

in multi-turn dialogue scenarios, resulting in mis-
aligned responses that degrade user experience.
For example, in a food ordering service context:
•All user utterances like ”I don’t want it” share
the same top-level intent: Order Operation.
•However, identical phrasing carries fundamen-
tally different meanings across scenarios:
Using undifferentiated responses based solely
on top-level intent causes severe semantic misalign-
ment and poor user experience.
1. Order Operation - Order Modification (Adding dish):
Agent: Would you like to add a cheese stick? Second item
50
User: Then I don’t want it.
User Real Intent: Order Modification Operation →Abort
add-on
2. Order Operation - Order Cancellation (Payment confir-
mation):
Agent: Your total is ¥98. Please confirm payment.
User: Then I don’t want it.
User Real Intent: Order Cancellation Operation →Termi-
nate transaction
To resolve this, we implement a dual-layer intent
framework for knowledge graph construction:
•intent level1( I1): Defines primary intent cate-
gories to enhance recognition accuracy
•intent level2 ( I2): Specifies fine-grained sub-
intents under each I1to prevent ambiguity
Formally:
•Iassistant
1 :assistant intent level1
•Iassistant
2 :assistant intent level2
•Iuser
1:user intent level1
•Iuser
2:user intent level2
3.2.2 Conversation Node
In multi-turn dialogue scenarios, generating con-
textually appropriate responses is the most funda-
mental and critical capability. Beyond extracting
expert knowledge via our dual-layer intent graph,
we integrate Conversation RAG and GraphRAG to
synthesize goal achieved historical dialogues with
precise intents, thereby retrieving optimal response
strategies that better align with the user’s current
intent. As illustrated in the figure 2, each assistant-
level-2 intent (assistant intent level2) is linked tohigh-quality goal achieved History conversation
(D) under that intent. This addresses the limita-
tions of single-layer intent graph retrieval, which
may yield poor matches or suboptimal perfor-
mance. By associating assistant intent level2 with
exemplary historical experiences, our approach si-
multaneously achieves intent matching and identi-
fies optimal historical dialogues. This enables the
generation of candidate responses that integrate
refined intents with optimized response strategies
3.2.3 Graph Construction
Specifically, leveraging high-quality goal achieved
historical multi-turn dialogue data from human
experts, we process each conversation (spanning
multiple turns) as follows: For every dialogue turn,
an LLM identifies dual-layer intents for both par-
ticipants. A dialogue intent transition graph is then
constructed based on the sequential relationships
between these intents. Identical intents are merged
into single nodes, with edges weighted by the co-
occurrence frequency of connected nodes. Higher-
frequency intent nodes receive higher retrieval pri-
ority in subsequent operations. As shown in the fig-
ure, both assistant and user intents are decomposed
into Level-1 and Level-2 intents. Each Level-2
intent node links to exemplary historical response
strategies. To address the low retrieval efficiency
of querying separate Level-1 and Level-2 intent
subgraphs, we innovatively combine assistant and
user intents into unified Assistant-User Intent Pair
Nodes ( P), significantly enhancing retrieval per-
formance.
P(Iassistant
2 ,Iuser
2 ) =
(assistant intent level2,user intent level2,frequency )
(1)
3.3 Inference Phase
During the inference phase, we propose a dual-
layer retrieval mechanism that integrates intent
transition retrieval and semantic similarity retrieval.
This dual-layer approach enables mutually-assisted
adaptive retrieval, enhancing both retrieval effec-
tiveness and efficiency.
3.3.1 Intent-based Retrieval Path
The concrete steps for intent-based graph retrieval
are as follows:
1.Intent Identification: An LLM processes the
user’s current query to identify the user intent.

Figure 3: Dual-path inference architecture for dialogue
systems. The flow comprises: (1) an Intent-based Re-
trieval Path and (2) a Conversation-based Retrieval
Path. Results from both paths are aggregated through
Weighted Scoring prior to Response Generation.
Figure 4: Workflow of the Intent-based Retrieval Path
during inference. The flow derives the assistant-user
intent pair from the user query, retrieves candidate intent
paths using the Intent Graph, and subsequently selects
the best path by calculating the frequency of intent
occurrences. This path directly informs the response
generation, forming one core branch of the dual-path
system.
2.Intent Pair Generation: Combines Iuser
2 with the
preceding assistant intent Iassistant
2 to form an
Assistant-User Intent Pair.
3.Graph Retrieval: The system retrieves top-
ranked candidate response intents based on
the normalized frequency metric f′, where co-
occurrence frequencies are normalized for com-
parative purposes.
f′(Iassistant
2 ) =f(Iassistant
2 )
max( f(Iassistant
2 ))(2)
When no matching intent is found during re-
trieval, a fallback intent vectorization RAG ap-
proach is activated:
•Intent Pair Vectorization: Embed the Assistant-
User Intent Pair using the BGE-M3 model.
•Candidate Intent Pair Filtering: Filter candidate
intent pairs sharing identical Level-1 intent ( I1)
with the current P.
•Similarity Calculation: Compute cosine similar-
ity between the current Pand filtered candidates,
retaining pairs exceeding a predefined threshold.
The detailed intent recognition algorithm is pro-
vided in the Appendix.3.3.2 Conversation-based Retrieval Path
Figure 5: Workflow of the Conversation-based Retrieval
Path during inference. The flow matches the identified
assistant-user intent pair against historical conversations
stored in memory via embedding-based similarity re-
trieval. Retrieved candidates undergo weighted scoring
based on contextual relevance, ranking, and selection
of top examples to guide context-aware response gener-
ation.
Beyond intent-based retrieval for candidate in-
tents, we simultaneously integrate Conversation
RAG semantic retrieval within the Intent Graph,
implementing multi-path recall to enhance recall
rate. Specifically:
1.Embed the conversation node ( Dc) under the cur-
rent Assistant-User Intent Pair’s assistant-level-2
intent
2. Embed the current memory ( Dm)
3.Compute similarity between DcandDmembed-
dings using BGE-M3
4.Retrieve historical dialogues with highest simi-
larity to Dm
Embedding:
em=BGE-M3 (Dm) (3)
ec=BGE-M3 (Dc) (4)
Similarity:
Similarity (Dc, Dm) = cos( ec,em)(5)
3.3.3 Dual Weighted Scoring
Unlike traditional RAG systems that rely solely on
semantic similarity (Conversation RAG) or static
knowledge graphs (GraphRAG), our approach in-
novatively integrates the intent transition map with
semantic similarity retrieval. This dual-layer re-
trieval mechanism enables mutually-assisted adap-
tive retrieval, significantly enhancing retrieval ef-
fectiveness and efficiency. For each candidate As-
sistant Intent, a weighted score is computed:
HI=α·f′(Iassistant
2 )
+ (1−α)·Similarity (Dc, Dm)(6)

Retrieval:
Top-k D hist= arg max
I∈Iuser
2(HI)(7)
where αdenotes a configurable parameter gov-
erning the weight allocation between intent pat-
terns and semantic similarity. CID-GraphRAG
leverages the high-quality dialogue histories and
candidate intents selected through this process as
few-shot examples for subsequent response gen-
eration. This methodology effectively extracts ex-
pert human knowledge, substantially improving
the accuracy and fluency of assistant responses in
dialogues.
3.4 Response Generation
Figure 6: Composition framework for LLM response
generation prompts. Key components integrated into
the prompt include: Assistant Intent, Top-k Example
Responses, and Instructions, alongside the original User
Question and inferred User Intent.
The top-k exemplary goal achieved historical
dialogues and candidate intents (typically k=5)
are selected through weighted scoring during in-
ference. An LLM then generates responses us-
ing a structured prompt that extends traditional
templates by incorporating user intent and CID-
GraphRAG-retrieved top-k examples as few-shot
instances. The prompt components are illustrated
in Figure 6:
• User Question: Current user query
• Memory: Contextual dialogue history
• User Intent: LLM-identified user intent
•Few-Shot: Top-k exemplary goal achieved his-
torical dialogues retrieved via CID-GraphRAG’s
dual-layer retrieval
• Instructions: Response generation guidelines
4 Experiments
4.1 Experimental Setup
4.1.1 Dataset and Metrics
We conducted experiments on a real-world cus-
tomer service dataset consisting of 268 conversa-tions between representatives and drivers regarding
vehicle sticker issues. Following standard practice,
we partitioned the dataset into training ( 80%, 214
dialogues with 1,299 conversation turns), valida-
tion ( 10%, 27 dialogues with 149 turns), and test
(10%, 27 dialogues with 126 turns) sets.
We employed two complementary evaluation
approaches:
LLM-as-Judge Evaluation : We utilized
Claude 3.7 Sonnet as an independent evaluator
to assess both retrieved response quality and gen-
erated responses. The LLM compared outputs
from different methods based on relevance, help-
fulness, style consistency, contextual appropriate-
ness, and professional standards. Evaluations were
conducted blindly. For each comparison, LLM
provided numerical ratings (1-10) for each sys-
tem, with the highest-rated system(s) designated
as ”winners” for that example. When multiple sys-
tems received identical highest ratings, all were
counted as winners, resulting in ties.
Automatic Metrics : We calculated BLEU,
ROUGE, METEOR and BERTSCORE between
system outputs and ground truth responses, as well
as between retrieved responses and ground truth.
4.1.2 Baselines
We compare our CID-GraphRAG approach with
three distinct methods:
Direct LLM : Generates responses using only
the conversation history without any retrieval aug-
mentation.
Intent RAG : Retrieves examples based solely
on intent graph matching. This method identifies
current intents and the most frequent system intent
candidate from graph. Then it randomly selects 5
conversations as in-context examples.
Conversation RAG : Retrieves examples based
solely on semantic similarity of the conversation
history. This method encodes the entire conversa-
tion context and retrieves the top 5 semantically
similar conversations from the training set as in-
context examples.
4.1.3 Implementation Details
Response Generation : We generate output us-
ing Claude 3.7 Sonnet. For all RAG methods, we
perform few-shot prompting using five examples
(5-shot), where each shot represents one in-context
retrieved response used when prompting the model.
For the Direct LLM approach, Claude 3.7 Sonnet is
prompted with the conversation history and asked

Figure 7: Semantic Match vs. Exact Match Effective-
ness on Coverage-Limited Cases
to generate an appropriate response in a zero-shot
manner.
LLM Configuration : All experiments used
Claude 3.7 Sonnet with temperature = 0.0 for de-
terministic outputs.
4.2 Hyperparameter Analysis
Before presenting the main results, we analyzed
key hyperparameters to determine optimal configu-
rations for our approach.
4.2.1 Semantic Match vs. Exact Match for
Intent Pairing
We investigated the impact of semantic matching
versus exact matching for intent pairs. In the val-
idation set, 58 dialogue turns ( 39% of total) had
no exact intent pair matches in the intent graph but
had semantic matches available. Figure 1 presents
results for these challenging cases.
LLM-as-Judge results in retrieval quality show-
ing semantic matching consistently outperforms
exact matching across all tested methods
For Intent RAG, exact matching completely
failed in these 58 cases, while semantic matching
successfully retrieved relevant examples. Even for
CID-GraphRAG, semantic matching consistently
outperformed exact matching across all weight con-
figurations, highlighting the critical importance of
semantic intent matching in real-world conversa-
tion systems.
4.2.2 Weight Parameter Sensitivity Analysis
We examined five different weight configurations
for CID-GraphRAG on validation set to deter-
mine the optimal balance between intent-based and
semantic-based retrieval, where
α∈ {0.1,0.3,0.5,0.7,0.9}
Our empirical analysis reveals two key findings
regarding the parameter α: (1) the configurationTable 1: Automatic Metric Results in Retrieval Perfor-
mance Across All Weight Configurations
Configuration BLEU-2 ↑BLEU-4 ↑ROUGE-1 ↑ROUGE-2 ↑ROUGE-L ↑METEOR ↑BERTSCORE ↑
α=0.1 7.48 2.89 23.89 5.78 20.14 14.76 63.32
α=0.3 7.01 2.71 23.37 5.33 19.81 14.24 63.15
α=0.5 6.95 2.7 23.27 5.3 19.76 14.18 63.11
α=0.7 6.95 2.7 23.27 5.3 19.76 14.18 63.11
α=0.9 6.95 2.7 23.27 5.3 19.76 14.18 63.11
with minimal intent weight ( α= 0.1) consistently
achieves optimal performance across all evaluation
metrics, with performance degradation as intent
weight increases; and (2) weight configurations
withα≥0.5yield identical retrieval results, sug-
gesting a saturation threshold beyond which intent-
based retrieval dominates the ranking process. To
validate these automatic evaluation results, we per-
formed LLM-as-Judge evaluation and confirmed
the results are aligned with our findings.
These observations suggest two important in-
sights regarding the dual-pathway retrieval mecha-
nism. First, intent information provides a valuable
structural signal that complements semantic re-
trieval by disambiguating between utterances with
high lexical similarity but different conversational
functions. Second, excessive weighting of intent
information proves detrimental to system perfor-
mance, as it prioritizes conversational patterns over
semantic content, retrieving examples that main-
tain dialogue structure but potentially lack context
relevance.
The optimal balance appears to be a predomi-
nantly semantic-driven approach ( 90%) guided by
a small but significant intent component ( 10%) that
helps the system maintain awareness of the con-
versation’s structural dynamics. Based on these
findings, we adopt α= 0.1settings for our main
experiments.
4.3 Evaluation Results
We evaluate CID-GraphRAG against baseline ap-
proaches using two complementary methodologies:
automatic metrics and LLM-as-judge assessments.
Figure 8 presents the LLM-as-judge evaluation re-
sults, showing win counts for both retrieval quality
and response generation quality across all methods
on the 126 test cases. Table 2 and Table 3 show-
ing automatic metrics for retrieval examples and
generated response separately.
4.3.1 Retrieval Quality Evaluation
The retrieval quality evaluation examines how ef-
fectively each method identifies relevant examples
to support response generation.

Figure 8: LLM-as-Judge Evaluation Results.LLM-as-
judge evaluation results comparing win counts for both
retrieval quality (left bars) and response generation qual-
ity (right bars) across all methods.
Table 2: Retrieval Quality Comparison Across Auto-
matic Metrics. Automatic metric scores comparing
retrieved examples with ground truth responses. Ar-
rows ( ↑) indicate higher scores are better. Bold values
represent the best performance for each metric
Model BLEU-2 ↑BLEU-4 ↑ROUGE-1 ↑ROUGE-2 ↑ROUGE-L ↑METEOR ↑BERTSCORE ↑
Intent RAG 6.47 2.26 22.67 4.3 19.08 15.31 63.10
Conversation RAG 8.22 3.40 24.60 6.00 21.06 17.02 63.74
CID-GraphRAG 8.74 3.50 26.09 6.52 22.15 18.21 64.70
As shown in Table 2, CID-GraphRAG consis-
tently outperforms baseline methods across all au-
tomatic metrics, demonstrating its superior abil-
ity to retrieve relevant information. The LLM-as-
judge evaluation for retrieval quality (left bars in
Figure 2) further confirms this superiority. These
observations indicates that combining intent transi-
tions with semantic similarity leads to more rele-
vant retrieved examples.
4.3.2 Response Generation Evaluation
The response generation evaluation assesses the
quality of final system outputs after augmentation
with retrieved examples.
Table 3: Response Generation Quality Comparison
Across Automatic Metrics. Automatic metric scores
comparing generated responses with ground truth across
all methods. Arrows ( ↑) indicate higher scores are bet-
ter. Bold values represent the best performance for each
metric
Model BLEU-2 ↑BLEU-4 ↑ROUGE-1 ↑ROUGE-2 ↑ROUGE-L ↑METEOR ↑BERTSCORE ↑
Direct LLM 6.15 1.46 22.53 3.25 18.25 18.13 62.54
Intent RAG 6.62 1.56 23.97 3.88 19.66 20.48 63.32
Conversation RAG 7.85 1.85 25.35 5.08 20.03 21.32 63.91
CID-GraphRAG 8.33 2.06 26.22 5.33 21.01 22.58 64.41
The response generation results in Table 3
establish a clear performance hierarchy: CID-
GraphRAG >Conversation RAG >Intent RAG
>Direct LLM. CID-GraphRAG maintains its ad-
vantage across all automatic metrics.The right bars in Figure 2 show the LLM-as-
judge results for response generation quality. CID-
GraphRAG leads with 60 wins, compared to 38
wins for Conversation RAG, 37 wins for Direct
LLM, and 23 wins for Intent RAG.
A notable observation is that Conversation RAG
(38 wins) and Direct LLM (37 wins) perform
nearly identically despite the former utilizing re-
trieval augmentation. Adding intent information to
create CID-GraphRAG results in 60 wins, achieved
a relative improvement of 58% compared to Con-
versation RAG, highlighting the powerful synergis-
tic effect of combining even a small intent compo-
nent with semantic retrieval.
4.4 Discussion
4.4.1 Effectiveness of CID-GraphRAG
Our experimental results consistently demonstrate
the superiority of CID-GraphRAG over other con-
ventional RAG approaches with two key insights:
Guided retrieval improves example quality :
Despite allocating only a 0.1 weight to intent in-
formation, CID-GraphRAG showed a substantial
improvement in win count over Conversation RAG
in response generation. Intent information acts as
a powerful guiding mechanism that helps filter out
semantically similar but contextually inappropriate
examples, reducing semantic drift in the retrieval
process. This helps distinguish between superfi-
cially similar utterances that serve different con-
versational purposes, providing domain-specific
structure to otherwise domain-general semantic
embeddings.
Robustness to coverage gaps : The semantic
matching approach successfully addresses cover-
age limitations in the intent graph. In cases where
exact intent matches were unavailable, the system
could still identify relevant examples through se-
mantic similarity, demonstrating the complemen-
tary nature of our dual retrieval approach.
4.4.2 Limitations and Future Work
While CID-GraphRAG shows promising results,
several limitations and future directions are worth
noting:
Reinforcement Learning Integration : The cur-
rent framework relies on supervised learning from
goal achieved dialogues to construct intent transi-
tion graphs. A promising future direction is inte-
grating reinforcement learning techniques to dy-
namically optimize intent transition probabilities

and retrieval weighting parameters based on con-
versation outcomes. This could enable the system
to continuously improve its retrieval strategy.
Domain adaptability : Our current evaluation
focuses on a specific domain of vehicle sticker-
related customer service. Future work should sys-
tematically investigate the framework’s transfer-
ability to broader multi-turn dialogue scenarios, in-
cluding complex task-oriented dialogues and cross-
domain interactions where conversations span mul-
tiple service areas.
5 Conclusion
In conclusion, to address the challenges of poor
performance in multi-turn customer service dia-
logue systems, we propose CID-GraphRAG, a
novel framework that integrates intent-driven graph
structures with semantic similarity retrieval mech-
anisms. Our approach leverages knowledge graphs
to enhance reasoning capabilities and enable more
precise retrieval, while systematically incorporat-
ing dialogue history to improve contextual under-
standing and response coherence, thereby signifi-
cantly boosting answer quality and relevance.
Experimental results demonstrate that CID-
GraphRAG significantly outperforms both
semantic-only and intent-only baseline approaches
across all evaluation criteria, achieving a 58%
relative improvement in response quality over
conventional Conversation RAG methods ac-
cording to LLM-as-judge evaluations, including
relevance, practicality, language style matching,
contextual adaptability, and response uniqueness.
These findings establish CID-GraphRAG as an
effective framework for advancing conversational
AI in knowledge-intensive domains where both
contextual relevance and goal-oriented progression
are essential. In the future, we will explore
the integration of reinforcement learning with
CID-GraphRAG and extend its applicability
to broader multi-turn dialogue scenarios (e.g.,
complex task-oriented dialogues and cross-domain
interactions).
References
Garima Agrawal, Sashank Gummuluri, and Cosimo
Spera. 2024. Beyond-rag: Question identification
and answer generation in real-time conversations.
Preprint , arXiv:2410.10136. ArXiv:2410.10136v1.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, andHannaneh Hajishirzi. 2023. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graphrag approach to query-focused summarization.
Accessed: 2025-06-17.
Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chu-
laka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj
Shah, Huaiyu Zhu, Danish Contractor, and Marina
Danilevsky. 2025. Mtrag: A multi-turn conversa-
tional benchmark for evaluating retrieval-augmented
generation systems. Preprint , arXiv:2501.03468.
Tzu-Lin Kuo, Feng-Ting Liao, Mu-Wei Hsieh, Fu-
Chieh Chang, Po-Chun Hsu, and Da-Shan Shiu.
2024. Rad-bench: Evaluating large language mod-
els’ capabilities in retrieval augmented dialogues.
Preprint , arXiv:2409.12558.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Preprint ,
arXiv:2005.11401.
Jasper Linders and Jakub M. Tomczak. 2025. Knowl-
edge graph-extended retrieval augmented generation
for question answering. Preprint , arXiv:2504.08893.
Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu
Lee, Mohammad Shoeybi, and Bryan Catanzaro.
2024. Chatqa: Surpassing gpt-4 on conversational
qa and rag.
ResearchGate. 2025. Learning when to retrieve, what
to rewrite, and how to respond in conversational qa.
Ran Xu, Wenqi Shi, Yuchen Zhuang, Yue Yu, Joyce C.
Ho, Haoyu Wang, and Carl Yang. 2025. Collab-rag:
Boosting retrieval-augmented generation for com-
plex question answering via white-box and black-box
llm collaboration.
Zhentao Xu, Mark Jerome Cruz, Matthew Guevara,
Tie Wang, Manasi Deshpande, Xiaofeng Wang, and
Zheng Li. 2024. Retrieval-augmented generation
with knowledge graphs for customer service question
answering. Preprint , arXiv:2404.17723.
Linhao Ye, Zhikai Lei, Jianghao Yin, Qin Chen,
Jie Zhou, and Liang He. 2024. Boosting con-
versational question answering with fine-grained
retrieval-augmentation and self-check. Preprint ,
arXiv:2403.18243.
Chi Zhang, Zhizhong Tan, Xingxing Yang, Weiping
Deng, and Wenyong Wang. 2025a. Credible plan-
driven rag method for multi-hop question answering.
Preprint , arXiv:2504.16787.

Feiyuan Zhang, Dezhi Zhu, James Ming, Yilun Jin,
Di Chai, Liu Yang, Han Tian, Zhaoxin Fan, and Kai
Chen. 2025b. Dh-rag: A dynamic historical context-
powered retrieval-augmented generation method for
multi-turn dialogue. Preprint , arXiv:2502.13847.
Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye,
Xinghe Chen, and Siqiang Luo. 2025. Graph-
based approaches and functionalities in retrieval-
augmented generation: A comprehensive survey. Ac-
cessed: 2025-06-17.

A Appendix
A.1 Intent Classification Code
Algorithm 1: Two-Stage Intent Classification for Dialogue Turns
Input: Dialogue turn Tcontaining current utterances and historical dialogue context
Output: System intents (Iassistant
1 , Iassistant
2 ), User intents (Iuser
1, Iuser
2)
foreach participant Pin{assistant, user }do
Stage 1 - Primary Intent Classification: ;
Construct a prompt containing:;
- Domain context;
- Historical dialogue context;
- Current dialogue turn;
- Comprehensive list of primary intent categories with descriptions;
- 3-5 few-shot examples showing correct primary intent classification;
Send prompt to Claude 3.7 Sonnet;
Extract primary intent IP
1from structured XML response;
Stage 2 - Secondary Intent Classification: ;
Construct a prompt containing:;
- Domain context;
- Historical dialogue context;
- Current dialogue turn;
- Primary intent IP
1determined in Stage 1;
- List of candidate secondary intents specific to the identified primary intent;
- 3-5 few-shot examples showing correct secondary intent classification;
Send prompt to Claude 3.7 Sonnet;
Extract secondary intent IP
2from structured XML response;
end
return (Iassistant
1 , Iassistant
2 , Iuser
1, Iuser
2)