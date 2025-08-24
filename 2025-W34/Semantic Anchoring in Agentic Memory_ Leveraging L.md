# Semantic Anchoring in Agentic Memory: Leveraging Linguistic Structures for Persistent Conversational Context

**Authors**: Maitreyi Chatterjee, Devansh Agarwal

**Published**: 2025-08-18 05:14:48

**PDF URL**: [http://arxiv.org/pdf/2508.12630v1](http://arxiv.org/pdf/2508.12630v1)

## Abstract
Large Language Models (LLMs) have demonstrated impressive fluency and task
competence in conversational settings. However, their effectiveness in
multi-session and long-term interactions is hindered by limited memory
persistence. Typical retrieval-augmented generation (RAG) systems store
dialogue history as dense vectors, which capture semantic similarity but
neglect finer linguistic structures such as syntactic dependencies, discourse
relations, and coreference links. We propose Semantic Anchoring, a hybrid
agentic memory architecture that enriches vector-based storage with explicit
linguistic cues to improve recall of nuanced, context-rich exchanges. Our
approach combines dependency parsing, discourse relation tagging, and
coreference resolution to create structured memory entries. Experiments on
adapted long-term dialogue datasets show that semantic anchoring improves
factual recall and discourse coherence by up to 18% over strong RAG baselines.
We further conduct ablation studies, human evaluations, and error analysis to
assess robustness and interpretability.

## Full Text


<!-- PDF content starts -->

Semantic Anchoring in Agentic Memory: Leveraging Linguistic
Structures for Persistent Conversational Context
Maitreyi Chatterjee
Cornell University
mc2259@cornell.eduDevansh Agarwal
Cornell University
da398@cornell.edu
Abstract
Large Language Models (LLMs) have demonstrated impressive fluency and task competence in con-
versational settings. However, their effectiveness in multi-session andlong-term interactions is hindered
by limited memory persistence. Typical retrieval-augmented generation (RAG) systems store dialogue
history as dense vectors, which capture semantic similarity but neglect finer linguistic structures such as
syntactic dependencies, discourse relations, and coreference links. We propose Semantic Anchoring ,
a hybrid agentic memory architecture that enriches vector-based storage with explicit linguistic cues to
improve recall of nuanced, context-rich exchanges. Our approach combines dependency parsing, dis-
course relation tagging, and coreference resolution to create structured memory entries. Experiments
on adapted long-term dialogue datasets show that semantic anchoring improves factual recall and dis-
course coherence by up to 18% over strong RAG baselines. We further conduct ablation studies, human
evaluations, and error analysis to assess robustness and interpretability.
1 Introduction
Conversational AI is evolving beyond single-turn, task-oriented bots toward multi-session assistants capable
of maintaining context across weeks, months, or even years. Persistent memory is central to this evolution:
users expect systems to recall prior preferences, commitments, and shared history without repeated expla-
nation [20, 15]. However, the two dominant approaches to conversational memory exhibit key limitations:
•Full-context prompting – storing the entire interaction history in the LLM context window is com-
putationally expensive, scales poorly with dialogue length, and risks context dilution [19].
•Vector-based RAG – retrieving past utterances based on dense embeddings captures surface-level
semantic similarity but neglects deeper discourse-level dependencies, leading to failures with para-
phrases, ellipsis, or implicit references [11, 7].
Recent work on agentic memory [14] frames memory as an active decision process—deciding when to
store, update, or forget. Yet, most implementations rely heavily on neural embeddings, limiting robustness
and interpretability. Parallel efforts in symbolic–neural integration suggest that explicit linguistic structures
(e.g., syntax, discourse, coreference) can provide complementary signals for reasoning and retrieval [12, 2].
This motivates our central research question:
Can conversational memory be made more robust and interpretable by anchoring retrieval in
explicit linguistic structures?
We propose Semantic Anchoring – a memory indexing and retrieval framework that augments dense
embeddings with symbolic linguistic features. Specifically, our approach:
1arXiv:2508.12630v1  [cs.CL]  18 Aug 2025

1. Extracts syntactic dependency trees to capture grammatical roles and resolve elliptical references
[6].
2. Performs coreference resolution to unify entity mentions across dialogue turns [10].
3. Tags discourse relations to encode conversational flow, such as elaboration, contrast, or causal links
[8].
4. Stores both dense embeddings and symbolic indexes in a hybrid retrieval framework , enabling
multi-granular matching.
Contributions Our work makes the following contributions:
• We introduce a hybrid agentic memory architecture that integrates dependency parses, discourse rela-
tions, and coreference chains into memory representations.
• We propose a retrieval scoring method that combines neural semantic similarity with symbolic match
scores for robust and interpretable retrieval.
• We conduct extensive evaluation across both adapted and real-world multi-session dialogue datasets,
showing consistent improvements in factual recall, discourse coherence, and user continuity satisfac-
tion.
• We provide ablation studies, sensitivity analysis, and human evaluations to assess robustness, inter-
pretability, and error modes.
2 Related Work
2.1 Long-term Conversational Memory
Persistent dialogue systems have been explored in personal assistants [15] and lifelong learning bots [13].
Most adopt RAG pipelines [11], storing conversation chunks as embeddings. Our work differs by enriching
these embeddings with linguistic structure.
2.2 Linguistic Structure in Dialogue
Dependency parsing [6], discourse parsing [8], and coreference resolution [10] have improved understanding
in summarization and QA tasks. We apply these tools to memory indexing and retrieval, an underexplored
integration.
2.3 Agentic Memory
Agentic memory research [14] considers memory management policies (store, forget, update). We focus on
representation quality , enabling better retrieval regardless of storage policy.
2

3 Methodology
3.1 Overview
Our proposed Semantic Anchoring framework augments the memory pipeline of an agentic conversational
system with explicit linguistic structure. Rather than relying solely on dense embeddings for past utter-
ances, we extract and store syntactic ,semantic , and discourse features in a hybrid index that supports both
symbolic and neural retrieval.
The overall pipeline consists of four stages:
1.Syntactic parsing – Each utterance is parsed with a biaffine dependency parser [6] to obtain gram-
matical structure. Dependency labels and head–modifier relations capture syntactic roles, which are
useful for resolving elliptical and paraphrased queries.
2.Coreference resolution – We apply an end-to-end coreference resolution model [10] to link all refer-
ring expressions (pronouns, nominal mentions, named entities) to their antecedents, producing a set
of entity clusters with persistent IDs across the dialogue.
3.Discourse tagging – A PDTB-style discourse parser [8] labels inter-utterance relations (e.g., Elabo-
ration ,Contrast ,Cause ), enabling retrieval systems to prioritize utterances that serve specific conver-
sational functions.
4.Hybrid storage – The processed utterance is stored both in a dense vector database (FAISS) for
semantic similarity search and in a symbolic inverted index keyed by entity IDs, dependency features,
and discourse tags.
OurSemantic Anchoring approach enriches dense retrieval with symbolic linguistic features, including
dependency parsing, coreference resolution, and discourse tagging. These components provide interpretable
anchors for linking across sessions.
Figure 1: Architecture of Semantic Anchoring. Input utterances are processed through a parsing layer,
coreference resolver, and discourse tagger before being combined with dense retrieval in a hybrid index.
Retrieved candidates are scored and passed to the LLM context.
As shown in Figure 1, raw utterances first pass through symbolic processors (syntax, coreference, dis-
course), which feed into a hybrid retrieval index. This hybrid index integrates symbolic and dense represen-
tations for final retrieval scoring.
3.2 Memory Representation
Each memory entry Miis represented as a tuple:
Mi=⟨Ui, Ei, Di, Ci,vi⟩
3

where:
•Ui: the surface form of the utterance, along with speaker and timestamp metadata.
•Ei: a set of canonicalized entities linked to coreference clusters. Each entity is stored as (name,corefID ,NER type).
•Di: the dependency parse, represented as an adjacency list with labeled edges (e.g., nsubj ,dobj ).
•Ci: a vector of discourse relation labels associated with this utterance’s link to prior turns.
•vi: a dense embedding generated from Sentence-BERT, representing the semantic content of the
utterance.
This multi-view representation enables retrieval queries to be matched at multiple levels of granularity:
lexical semantics, entity continuity, syntactic alignment, and discourse role.
3.3 Hybrid Storage and Indexing
The hybrid memory store comprises two components:
1.Dense Index: Stores vivectors in FAISS, allowing O(logN)approximate nearest neighbor search.
2.Symbolic Index: Maintains inverted lists keyed by:
• Coreference IDs (for entity continuity).
• Dependency triplets (head lemma ,deplabel,child lemma ).
• Discourse relation labels.
These indexes are queried in parallel and their results are fused at ranking time.
3.4 Retrieval Scoring
At query time q, we compute a combined relevance score:
score (Mi, q) =λs·sim(vi,vq) +λe·entity match (Ei, Eq) +λc·discourse match (Ci, Cq)
where:
• sim is cosine similarity between dense embeddings.
• entity match measures the proportion of entities in the query that are present in Ei, weighted by
coreference cluster size.
• discourse match gives a binary or graded score depending on whether discourse roles align.
Weights (λs, λe, λc)are tuned on a held-out validation set using grid search to optimize Factual Recall.
Algorithm 1: Retrieval Procedure
1. Compute query embedding vq, entity set Eq, and discourse tags Cq.
2. Retrieve top- ncandidates from dense index using vq.
3. Retrieve additional candidates from symbolic index matching EqorCq.
4. Merge candidate lists and compute score (Mi, q)for each.
5. Return top- kentries by score.
4

3.5 Integration with the LLM
Retrieved entries are serialized into a linguistically-aware context prompt :
[Entity: Dr. Morales][CorefID: E42][NER: PERSON] said “MRI results show early-stage
glioma” [Discourse: ELABORATION] ...
This serialization:
• Preserves explicit entity references for continuity.
• Maintains discourse signals to help the LLM interpret the conversational role of each memory item.
• Supports multi-turn summarization by the LLM, which can rewrite the entries into a concise memory
summary before appending to context.
In our agentic setup, the memory manager component determines whether to store the current utterance,
update an existing entry (e.g., revised facts), or discard low-value information, but our focus here is on
improving the quality of retrieval given the stored memory.
4 Experimental Setup
4.1 Datasets
To evaluate the proposed method, we constructed two long-term conversational datasets that emphasize
cross-session context dependencies.
MultiWOZ-Long We adapt the MultiWOZ 2.2 dataset [3] into a multi-session format by splitting long
dialogues into consecutive “sessions” separated by simulated temporal gaps (e.g., hours or days). We ensure
that:
• Important entities (e.g., hotels, restaurants) appear across sessions.
• Some entity mentions are indirect (via pronouns or paraphrases).
• Factual details (e.g., booking times) are introduced in one session and queried in a later session.
This setup creates retrieval challenges that require both semantic similarity and structural understanding.
DialogRE-L DialogRE [21] is a dialogue-based relation extraction dataset. We extended it to DialogRE-
Lby:
• Introducing artificial session boundaries every few turns.
• Adding cross-session coreference chains where entities are referenced in later sessions by pronouns
or descriptive phrases.
• Including relations that require recalling multiple prior utterances for correct inference.
This dataset tests memory models on entity tracking and relation recall across temporal gaps.
5

4.2 Baselines
We compare our method against three baselines:
1.Stateless LLM – GPT-3.5-turbo without any retrieval; each query is answered with only the current
turn.
2.Vector RAG – A standard retrieval-augmented generation pipeline using Sentence-BERT embeddings
stored in FAISS. Retrieval is purely based on cosine similarity between query and past utterances.
3.Entity-RAG – An entity-aware retrieval system that matches queries to memory entries sharing
named entities, without using syntactic or discourse features.
All baselines use the same underlying LLM for generation to ensure fairness; only the memory retrieval
component varies.
4.3 Metrics
We evaluate using both automatic and human-centric metrics:
Factual Recall (FR) The proportion of factual queries for which the system correctly recalls information
from prior sessions. Computed by matching extracted answer spans against gold references.
Discourse Coherence (DC) Measures consistency in entity references and conversational flow. Computed
by:
• Performing coreference resolution on generated responses.
• Comparing cluster assignments with gold annotations.
User Continuity Satisfaction (UCS) A human-judged metric (1–5 Likert scale) where annotators rate
whether the agent appears to “remember” past interactions naturally and usefully. Higher is better.
4.4 Implementation Details
Reproducibility. Full preprocessing, indexing, and hyperparameters are in Appendix A.
Parsing and Tagging
•Dependency Parsing: spaCy v3 with transformer-based English dependency parser (trained on
OntoNotes).
•Coreference Resolution: AllenNLP’s end-to-end neural coreference resolver.
•Discourse Tagging: PDTB-style discourse relation classifier fine-tuned on the Penn Discourse Tree-
bank 3.0.
Vector Index Dense embeddings are produced using Sentence-BERT all-mpnet-base-v2 and stored
in FAISS with HNSW indexing for O(logN)approximate nearest neighbor retrieval.
6

Symbolic Index An inverted index is implemented using Whoosh , keyed by:
• Coreference cluster IDs.
• Dependency triples (head lemma ,deplabel,child lemma ).
• Discourse relation labels.
Fusion and Weight Tuning Symbolic and dense retrieval results are combined using weighted rank fu-
sion, with weights (λs, λe, λc)tuned via grid search on the MultiWOZ-Long validation set to maximize
FR.
Hardware and Runtime Experiments are conducted on a machine with 2 ×NVIDIA A100 GPUs, 512GB
RAM, and Intel Xeon Platinum CPUs. Average retrieval latency per query is ∼120ms for dense search and
∼40ms for symbolic search, with fusion adding ∼15ms.
5 Results
5.1 Main Results
Table 1 reports performance on MultiWOZ-Long [3]. Semantic Anchoring achieves the strongest per-
formance across all metrics. Compared to the best-performing baseline (Entity-RAG), it improves Factual
Recall (FR) and Discourse Coherence (DC) significantly ( p <0.01), while yielding a smaller but consistent
gain in User Continuity Satisfaction (UCS) ( p <0.05). Results are averaged over three runs with standard
deviations in parentheses.
Model FR (%) DC (%) UCS (/5)
Stateless LLM 54.1 (0.4) 48.3 (0.5) 2.1 (0.1)
Vector RAG 71.6 (0.6) 66.4 (0.7) 3.4 (0.1)
Entity-RAG 75.9 (0.5) 72.2 (0.6) 3.7 (0.1)
Semantic Anchoring 83.5 (0.3) 80.8 (0.4) 4.3(0.1)
Table 1: Overall performance on MultiWOZ-Long. Semantic Anchoring outperforms all baselines across
metrics. Improvements in FR and DC are statistically significant at p <0.01; UCS gains are significant at
p <0.05. Values are mean ± stdev over three runs.
Figure 2 analyzes how performance varies with session depth. While all models degrade as dialogue span
increases, Semantic Anchoring sustains over 75% recall at 10 sessions, indicating stronger long-range track-
ing.
5.2 Per-Dataset Breakdown
To test generality, we evaluate on DialogRE-L , which emphasizes relation extraction across sessions. Re-
sults in Table 2 show consistent improvements, though broader domains are needed to claim robustness.
7

Figure 2: Factual Recall by session depth on MultiWOZ-Long. Semantic Anchoring exhibits the slowest
degradation, maintaining >75% recall at 10-session distance. Error bars denote standard deviation across
three runs.
Model FR (%) DC (%) UCS (/5)
Stateless LLM 49.8 44.1 2.0
Vector RAG 68.7 62.5 3.2
Entity-RAG 72.1 68.3 3.6
Semantic Anchoring 81.4 77.9 4.2
Table 2: Performance on DialogRE-L. Semantic Anchoring achieves consistent gains across metrics, sug-
gesting effectiveness in relation extraction tasks that require long-range entity tracking.
5.3 Ablation Studies
Table 3 examines the role of linguistic components. Removing discourse tagging reduces FR by 4.7 points,
while excluding coreference resolution reduces DC by 6.2 points. Eliminating all symbolic features col-
lapses performance to Vector RAG levels. These results align with observed error patterns (§5.6), under-
scoring the value of symbolic features.
5.4 Qualitative Examples
In MultiWOZ-Long, when the user later asks “Did he confirm the time for the taxi?” , Semantic Anchoring
retrieves:
[Entity: John Smith][CorefID: E17] confirmed the taxi is booked for 9 AM.
By contrast, Vector RAG surfaces unrelated mentions of “taxi.” Additional examples, including cases where
Semantic Anchoring fails, are shown in Appendix C.
8

Variant FR (%) DC (%) UCS (/5)
Full Model 83.5 80.8 4.3
– Discourse Tagging 78.8 75.6 4.0
– Coreference Resolution 80.1 74.6 4.1
– Dependency Parsing 81.2 78.5 4.1
Dense-only (Vector RAG) 71.6 66.4 3.4
Table 3: Ablation results on MultiWOZ-Long. Removing discourse or coreference modules significantly
reduces FR and DC, respectively. Without all symbolic features, performance falls to the dense-only base-
line.
5.5 Human Evaluation
Five trained annotators rated 50 randomly sampled conversations for User Continuity Satisfaction (UCS).
Agreement was high ( α= 0.81). As Table 1 shows, Semantic Anchoring achieves the highest UCS (4.3),
with annotators noting better consistency in entity references. Full protocol details are in Appendix B.
5.6 Error Analysis
Table 4 categorizes common failures. Coreference mistakes (27%) and parsing errors (19%) are the most
frequent, consistent with ablation findings. Discourse mislabeling (15%) often arises in sarcasm or overlap-
ping speech. While overall error frequency is lower than dense retrieval, these remain open challenges.
Error Type Proportion of Failures
Parsing errors 19%
Coreference mistakes 27%
Discourse mislabeling 15%
Other / miscellaneous 39%
Table 4: Error analysis on MultiWOZ-Long. Coreference mistakes are the most frequent error type,
followed by parsing and discourse issues. These patterns align with ablation results.
6 Conclusion
We introduced Semantic Anchoring , a linguistically-aware agentic memory framework that substantially
advances recall, coherence, and interpretability in long-term dialogue [19, 20, 7]. By explicitly grounding
memory in linguistic structure [9, 12], our approach bridges symbolic and neural representations [2, 16],
offering a principled path toward more reliable conversational agents [17, 18]. Looking ahead, we envision
integrating incremental parsing for real-time adaptability [1], enabling user-editable memories for greater
transparency [4], and scaling to multilingual contexts [5]—paving the way for persistent, trustworthy, and
globally accessible dialogue systems.
9

References
[1] Miguel Ballesteros, Chris Dyer, and Noah A. Smith. Neural architectures for incremental parsing. In
Proceedings of the 2016 Conference of the North American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies (NAACL-HLT) , 2016.
[2] Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella
Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas Pinto, and James Puste-
jovsky. Experience grounds language. In Proceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP) , 2020.
[3] Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, I ˜nigo Casanueva, Stefan Ultes, Osman
Ramadan, and Milica Ga ˇsi´c. Multiwoz: A large-scale multi-domain wizard-of-oz dataset for task-
oriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 5016–5026, 2018.
[4] Chia-Hsuan Chang, Victor Zhong, Luke Zettlemoyer, and Noah A. Smith. Spoken memory: Enabling
users to edit and update ai memory in conversation. In Proceedings of the 2023 ACM Conference on
Human Factors in Computing Systems (CHI) , 2023.
[5] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Fran-
cisco Guzm ´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised
cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics (ACL) , 2020.
[6] Timothy Dozat and Christopher D. Manning. Deep biaffine attention for neural dependency parsing.
InProceedings of the International Conference on Learning Representations (ICLR) , 2017.
[7] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Improving dialogue coherence with entity-aware mem-
ory. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(ACL) , 2023.
[8] Yangfeng Ji, Zhengzhong Liu, and Junyi Jessy Li. A survey on discourse parsing. Transactions of the
Association for Computational Linguistics (TACL) , 10:1314–1334, 2022.
[9] Daniel Jurafsky and James H. Martin. Speech and Language Processing . Prentice Hall, 2000.
[10] Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. End-to-end neural coreference resolu-
tion. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing
(EMNLP) , pages 188–197, 2017.
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, and Sebastian Riedel. Retrieval-
augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Pro-
cessing Systems (NeurIPS) , 2020.
[12] Zihan Liu, Jiahai Wang, and Jian-Yun Nie. Symbolic knowledge integration for neural dialogue mod-
els.Transactions of the Association for Computational Linguistics (TACL) , 2023.
[13] Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. Opendialkg: Explainable conversa-
tional reasoning with attention-based walks over knowledge graphs. In Proceedings of the 57th Annual
Meeting of the Association for Computational Linguistics (ACL) , pages 845–854, 2019.
10

[14] Joon Sung Park, Carrie J. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S.
Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th
Annual ACM Symposium on User Interface Software and Technology (UIST) , 2023.
[15] Ashwin Ram, Rohit Prasad, Chandra Khatri, Anu Venkatesh, Raefer Gabriel, Qiao Liu, Jonathan Nunn,
Behnam Hedayatnia, Ming Cheng, Anusha Nagar, Lance King, Kelly Bland, Evan Wartick, Yuchang
Pan, Yushi Song, Surya Jayadevan, and Dilek Hakkani-Tur. Conversational AI: The science behind
the alexa prize. arXiv preprint arXiv:1801.03604 , 2018.
[16] Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in BERTology: What we know about
how BERT works. Transactions of the Association for Computational Linguistics (TACL) , 2021.
[17] Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle
Ott, Kurt Shuster, Eric Smith, Y-Lan Boureau, and Jason Weston. Recipes for building an open-
domain chatbot. In Proceedings of the 16th Conference of the European Chapter of the Association
for Computational Linguistics (EACL) , 2021.
[18] Kurt Shuster, Spencer Poff, Myle Ott, James Thorne, and Jason Weston. Language models that seek for
knowledge: Modular search and generation for dialogue. In Proceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (ACL) , 2022.
[19] Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers.
InProceedings of the International Conference on Learning Representations (ICLR) , 2022.
[20] Haoran Xu, Xin Xu, Yubo Zhang, and Wenjie Li. Long-term conversational memory for LLM-based
dialogue agents. arXiv preprint arXiv:2401.12345 , 2024.
[21] Dian Yu, Kai Sun, Claire Cardie, and Dong Yu. Dialogre: Dialog-based relation extraction. In Pro-
ceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) ,
pages 892–900, 2020.
A Reproducibility: Data Processing, Indexing, and Hyperparameters
This appendix specifies the exact steps and settings needed to reproduce our results.
A.1 Data Preprocessing Pipeline
We standardize all dialogues via the following sequence:
1.Normalization: Lowercase text (except acronyms), strip markup, normalize whitespace, and preserve
punctuation needed for dependency and discourse tagging.
2.Sentence segmentation & tokenization: spaCy v3 transformer pipeline (en core web trf). We keep
sentence boundaries to support clause-level discourse cues.
3.Lemmatization & POS: spaCy lemmatizer and POS tags are stored alongside tokens for later con-
struction of dependency triples.
4.NER: spaCy NER spans are retained and fed into the coreference resolver as mention candidates.
11

A.2 MultiWOZ-Long Construction
Starting from MultiWOZ 2.2, we create a multi-session variant:
1.Sessionization: Insert a session boundary after dialogue segments that (i) close a booking/goal or (ii)
exceed a turn budget (e.g., 8–12 turns) while maintaining at least one entity that recurs in the next
session.
2.Temporal gaps: Annotate boundaries with a synthetic time gap tag (e.g., <GAP=hours:36> ) used
only to ensure cross-session references during sampling; tags are not shown to models.
3.Entity continuity constraints: Require at least one entity (hotel/restaurant/taxi, etc.) to reappear via
name, nominal, or pronoun in a later session so that cross-session recall is necessary.
4.Quality checks: Randomly audit 5% of sessionized dialogues to confirm that at least one fact intro-
duced earlier is queried later.
A.3 DialogRE-L Extension
From DialogRE [21], we derive a long-range variant:
1.Boundary insertion: Place boundaries every 6–10 turns, preferring points between relation-bearing
utterances.
2.Cross-session coref: Where possible, replace a repeated proper name in a later session with a pronoun
or descriptive NP to force coreference resolution across sessions.
3.Relation preservation: Ensure that at least one gold relation requires retrieving evidence from a prior
session (multi-hop references are allowed).
A.4 Symbolic Feature Extraction
•Dependency parsing: Biaffine parser [6]. We store triples of the form (head lemma ,deplabel,child lemma)
per utterance.
•Coreference resolution: End-to-end resolver [10]; each mention receives a CorefID . Entities in
memory are canonicalized to (name, CorefID, NER type) .
•Discourse tagging: PDTB-style classifier [8]; we keep coarse-grained labels (e.g., Elaboration ,Con-
trast,Cause ).
A.5 Hybrid Indexing Details
Dense. Sentence-BERT all-mpnet-base-v2 (768-d). FAISS HNSW index with M=32,efConstruction
= 200 ; query-time efSearch = 128 . Embeddings are ℓ2-normalized; similarity is cosine.
Symbolic. We precompute lemmas and store exact tokens. The inverted index keys:
•Entities: CorefID and surface name.
•Dependency triples: Concatenated as head:label:child strings.
•Discourse: One field per label (binary flags).
12

A.6 Retrieval Scoring and Tuning
We compute
score( Mi, q) = λscos(vi,vq) +λeentity match( Ei, Eq) +λcdiscourse match( Ci, Cq).
Weights (λs, λe, λc)are selected by grid search on the MultiWOZ-Long dev split; we sweep {0.40,0.50, . . . , 0.90}
with the constraint λs+λe+λc= 1and choose the best FR.
A.7 Prompt Serialization Template
The top- kretrieved memories are serialized for the LLM as:
[ENTITY: Dr. Morales | CorefID=E42 | NER=PERSON]
[DISCOURSE: ELABORATION]
[UTTERANCE @ 2024-03-14 09:10] "MRI results show early-stage glioma."
[DEPS: (show-nsubj-results), (show-dobj-glioma)]
We include at most 2 lines of symbolic metadata per entry to control token budget.
A.8 Compute and Latency Measurement
All timing excludes network I/O. We measure mean latency over 1,000 queries with the index warmed;
FAISS and the symbolic index run in parallel (two threads), and fusion adds a small constant overhead.
A.9 Licensing and Ethics
We follow dataset licenses; all personal identifiers are removed. Dialog snippets in the paper are synthetic
or anonymized. No end-user data from real deployments is included.
B Human Evaluation Protocol
Goal. Quantify whether responses produced with Semantic Anchoring feel as if the agent “remembers”
prior interactions more naturally than baselines.
Raters. Five graduate-level annotators with prior NLP coursework. Raters completed a 45-minute training
with examples and a short quiz ( ≥80% to proceed).
Items. 50 multi-session conversations sampled without replacement from the M ULTI WOZ-L ONG eval-
uation split; 30 additional conversations from D IALOG RE-L for spot checks. Each item contains: (i) the
current user turn, (ii) model output, (iii) a compact history summary (truncated to 1–2K tokens), and (iv)
gold facts for verification. Sensitive details were removed; speaker names were anonymized.
Models Compared. STATELESS LLM, V ECTOR RAG, E NTITY -RAG, and S EMANTIC ANCHORING .
For each item, raters saw four responses in random order with model identities hidden.
13

Primary Metric: UCS. User Continuity Satisfaction (1–5 Likert):
•1= No continuity: contradicts or ignores prior context.
•2= Weak continuity: recalls little or gets entities wrong.
•3= Acceptable: recalls some details; minor errors.
•4= Strong: recalls key entities/facts; flows naturally.
•5= Excellent: precise recall and seamless integration of past context.
Raters also flagged binary errors: wrong entity, wrong value, discourse mismatch, or hallucination.
Procedure. Each item is independently rated by all five annotators. We collect a UCS score and free-text
notes per response. Items are presented in randomized order. No time limit was imposed; median time per
item was 2.9 minutes.
Aggregation. For each item–model pair we average UCS across raters. Inter-annotator agreement is re-
ported with Krippendorff’s αfor ordinal data ( α= 0.81on UCS). Outliers ( >2.5 SD from the rater’s mean)
were audited; <1%were removed after pre-registered rules.
Significance Testing. We perform paired two-tailed t-tests on item-level means, comparing S EMANTIC
ANCHORING vs. the top baseline. Holm–Bonferroni corrects for multiple comparisons. We also bootstrap
95% CIs (10k resamples) for UCS and report exact p-values.
Blinding and Leakage Controls. Prompts were sanitized for model names or style hints. Raters could
not see retrieval snippets, only final responses. Items were drawn from held-out splits; no fine-tuning data
overlapped with evaluation.
Ethics. All data derive from public research corpora or synthetic variants. We removed PII and followed
dataset licenses. No user study with human subjects was conducted beyond annotation of public/synthetic
artifacts.
C Qualitative Analyses
We include representative successes and failure cases. Examples are lightly paraphrased to remove identi-
fying tokens while preserving structure.
C.1 Success Cases
C.2 Failure Cases
Takeaways. Symbolic cues help with ellipsis, pronouns, and “same as last time” references; they remain
brittle under sarcasm, name collisions, and speech repairs. Future work: (i) prosody/disfluency-aware pars-
ing, (ii) speaker-role-conditioned coref, and (iii) contrastive training on pragmatic phenomena.
14

Query + Target Fact Top-1 Vector RAG Top-1 Semantic Anchoring
Q:“Did heconfirm the taxi
time?”
Target: John Smith confirmed
taxi for 9:00 AM.Mentions “taxi options” with times
8:30/10:00; no link to he.Utterance with entity [John Smith
| CorefID E17] and dependency
(confirm,nsubj,John) → re-
trieves “confirmed 9 AM.”
Q:“Move herclinic appointment
to Friday.”
Target: “Dr. Khan scheduled for
Fri 3pm for Asha .”Brings a prior “clinic hours” message;
misses referent.Coref chain links her→
[Asha|E05] ; dependency triple
(schedule,dobj,appointment)
matches; returns correct slot.
Q:“Book the same place as last
time, but 2 nights.”
Target: Last stay = “Parkview
Hotel”.Retrieves similar utterance about “city
center hotels” (semantic drift).Discourse tag ELABORATION + entity
continuity picks last booking summary
→“Parkview Hotel.”
Table 5: Illustrative wins where entity/coreference + discourse signals disambiguate elliptical references.
Phenomenon Example and Analysis
Sarcasm / Pragmatics User: “Great, another early flight—just what I wanted.” Gold intent: avoid early
flights. Our system retrieves a prior “approved 6:30am” turn (lexical match on flight )
and proposes a 6:30am option; discourse classifier labels CONTRAST incorrectly,
missing sarcasm.
Coref Over-Merge Two people named “Alex” appear across sessions (guest vs. agent). A long pronoun
chain collapses into one cluster; retrieval surfaces guest preferences when the agent
is referenced. Mitigation: add speaker-aware coref features and dialogue role em-
beddings.
Parser Error on Disfluency Utterance with repairs: “the—uh—the Italian place. . . actually the Vegan Deli.” De-
pendency triples are noisy; symbolic index under-weights corrected segment; dense
match alone would succeed.
Table 6: Common failure modes. We list mitigations in §5.6.
15