# What Breaks Knowledge Graph based RAG? Empirical Insights into Reasoning under Incomplete Knowledge

**Authors**: Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Yuan He, Jiaoyan Chen, Evgeny Kharlamov, Steffen Staab

**Published**: 2025-08-11 10:55:06

**PDF URL**: [http://arxiv.org/pdf/2508.08344v1](http://arxiv.org/pdf/2508.08344v1)

## Abstract
Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) is an
increasingly explored approach for combining the reasoning capabilities of
large language models with the structured evidence of knowledge graphs.
However, current evaluation practices fall short: existing benchmarks often
include questions that can be directly answered using existing triples in KG,
making it unclear whether models perform reasoning or simply retrieve answers
directly. Moreover, inconsistent evaluation metrics and lenient answer matching
criteria further obscure meaningful comparisons. In this work, we introduce a
general method for constructing benchmarks, together with an evaluation
protocol, to systematically assess KG-RAG methods under knowledge
incompleteness. Our empirical results show that current KG-RAG methods have
limited reasoning ability under missing knowledge, often rely on internal
memorization, and exhibit varying degrees of generalization depending on their
design.

## Full Text


<!-- PDF content starts -->

What Breaks Knowledge Graph based RAG? Empirical Insights
into Reasoning under Incomplete Knowledge
Dongzhuoran Zhou1,2, Yuqicheng Zhu2,3, Xiaxia Wang4, Hongkuan Zhou2,3, Yuan He4,
Jiaoyan Chen5,Evgeny Kharlamov1,2, Steffen Staab3,7
1University of Oslo,2Bosch Center for AI,3University of Stuttgart,
4University of Oxford,5The University of Manchester,7University of Southampton
dongzhuoran.zhou@de.bosch.com
Abstract
Knowledge Graph-based Retrieval-Augmented
Generation (KG-RAG) is an increasingly ex-
plored approach for combining the reasoning
capabilities of large language models with the
structured evidence of knowledge graphs. How-
ever, current evaluation practices fall short: ex-
isting benchmarks often include questions that
can be directly answered using existing triples
in KG, making it unclear whether models per-
form reasoning or simply retrieve answers di-
rectly. Moreover, inconsistent evaluation met-
rics and lenient answer matching criteria further
obscure meaningful comparisons. In this work,
we introduce a general method for constructing
benchmarks, together with an evaluation proto-
col, to systematically assess KG-RAG methods
under knowledge incompleteness. Our empiri-
cal results show that current KG-RAG methods
have limited reasoning ability under missing
knowledge, often rely on internal memoriza-
tion, and exhibit varying degrees of generaliza-
tion depending on their design.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a widely adopted framework for enhancing
large language models (LLMs) by incorporating ex-
ternal knowledge through a retrieve-then-generate
paradigm (Khandelwal et al., 2020; Izacard and
Grave, 2021; Borgeaud et al., 2022; Ram et al.,
2023). By conditioning the generation on the re-
trieved documents, RAG enables LLMs to answer
questions or perform tasks using more comprehen-
sive and up-to-date knowledge than what is stored
in their parameters. To improve retrieval accuracy,
enable structured reasoning and support explana-
tion, a few recent works have proposed RAG meth-
ods based on Knowledge Graph (KG-RAG) (Han
et al., 2024; Peng et al., 2024). Most of them di-
rectly use existing knowledge graphs (KGs) (Luo
et al., 2024; Sun et al., 2024), while some of them
construct and extend structured knowledge fromunstructured documents (Fang et al., 2024). Such
KG-RAG systems are expected to be well-suited
for structured reasoning, where answers require
synthesizing information from multiple connected
facts.
Despite growing interest, current evaluation
practices for KG-RAG fall short in two key
ways. First, most existing benchmarks (Yih
et al., 2016; Talmor and Berant, 2018a) are
constructed on top of complete KGs, where
direct evidence supporting the answer is read-
ily available. For example, given the question
“Who is the brother of Justin Bieber? ”, the KG
contains the triple hasBrother(JustinBieber,
JaxonBieber) , allowing the system to answer
the question without reasoning. However, real-
world KGs are often incomplete, and answer-
ing such questions in practice may require rea-
soning over alternative paths, e.g., combining
hasParent(JustinBieber, JeremyBieber) and
hasChild(JeremyBieber, JaxonBieber) to in-
fer the sibling relationship. As a result, current
benchmarks do not assess whether KG-RAG meth-
ods can reason over missing knowledge or simply
retrieve answers directly from explicit evidence.
Second, evaluation metrics across KG-RAG
studies are often inconsistent and unreliable. Defi-
nitions of commonly used metrics such as accuracy
and Hits@1 vary across studies—some equate ac-
curacy with Hits@1, while others use accuracy as
a proxy for recall, or provide vague descriptions
altogether. In addition, many studies only adopt
overly permissive matching criteria, treating pre-
dictions as correct if they contain the gold answer
as a substring, regardless of correctness or context.
This results in inflated performance estimates and
makes meaningful comparison difficult.
In this work, we present a general method for
constructing benchmarks and evaluation protocols
to systematically assess KG-RAG methods under
knowledge incompleteness. Each question in ourarXiv:2508.08344v1  [cs.AI]  11 Aug 2025

benchmark is constructed such that it cannot be
answered using a single explicit triple. Instead, the
answer must be inferred by reasoning over alterna-
tive paths in the KG. To construct this benchmark,
we follow a two-step process: (1) we mine high-
confidence logical rules from the KG using a rule
mining algorithm, and (2) we generate natural lan-
guage questions based on rule groundings, ensuring
that the directly supporting triple is removed while
the remaining KG still contains sufficient evidence
to infer the answer.
Our empirical study on this benchmark reveals
several key limitations of current KG-RAG systems.
First, most models struggle to recover answers
when direct supporting facts are removed, high-
lighting their limited reasoning capacity. Second,
training-based methods (e.g., RoG, GNN-RAG)
show stronger robustness under KG incomplete-
ness compared to non-trained systems. Third, we
find that textual entity labels substantially boost
performance, suggesting that LLMs rely heavily
on internal memorization.
2 Related Work
2.1 Knowledge Graph Question Answering
(KGQA)
The KGQA task aims to answer natural language
questions using the KG G, where Gis represented
as a set of binary facts r(s, o), with rdenoting a
predicate and s, odenoting entities. The answer to
each question is one or more entities in G.
Approaches to KGQA can be broadly catego-
rized into the following two types: (1) Semantic
parsing-based methods (Yih et al., 2016; Gu et al.,
2021; Ye et al., 2021) translate questions into for-
mal executable queries (e.g., SPARQL or logical
forms), which are then executed over the KG to
retrieve the answer entities. These methods offer
high precision and interpretability, as the reasoning
process is explicitly encoded. However, they face
several challenges, including difficulty in under-
standing semantically and syntactically complex
questions, handling diverse and compositional log-
ical forms, and managing the large search space
involved in parsing multi-relation queries(Lan et al.,
2021). (2) Embedding-based methods (Yao et al.,
2019; Baek et al., 2023), by contrast, encode ques-
tions and entities into a shared vector space and
rank answer candidates based on embedding sim-
ilarity. While these methods are end-to-end train-
able and do not require annotated logical forms,they often struggle with multi-hop reasoning (Qiu
et al., 2020), lack interpretability (Biswas et al.,
2023), and exhibit high uncertainty in their predic-
tions (Zhu et al., 2024, 2025b,a).
2.2 KGQA with RAG
Recently, a new line of work has emerged that inte-
grates large language models more tightly into the
KG reasoning process. These KG-RAG methods
go beyond previous KGQA approaches by coupling
structured retrieval with generative reasoning ca-
pabilities of LLMs. RoG (Luo et al., 2024) adopts
a planning-retrieval-reasoning pipeline, where an
LLM generates relation paths as plans, retrieves
corresponding paths from the KG, and reasons over
them for answer generation. G-Retriever (He et al.,
2024) retrieves subgraphs via Prize-Collecting
Steiner Tree optimization and provides them to
LLMs as soft prompts or text prompts for question
answering. GNN-RAG (Mavromatis and Karypis,
2024) uses a GNN to select candidate answers and
retrieves shortest paths to them, which are verbal-
ized for LLM-based answer generation. PoG (Chen
et al., 2024) instead decomposes questions into sub-
goals and iteratively explores and self-corrects rea-
soning paths using a guidance-memory-reflection
mechanism. ToG (Sun et al., 2024) treats the
LLM as an agent that interactively explores and
aggregates evidence on KGs through iterative beam
search, StructGPT (Jiang et al., 2023) introduces an
interface-based framework that enables LLMs to it-
eratively extract relevant evidence from structured
data such as KGs and reason over the retrieved
information to answer complex questions.
2.3 KGQA Benchmarks
Existing benchmarks such as WebQSP (Yih et al.,
2016), CWQ (Talmor and Berant, 2018b), and
GrailQA (Gu et al., 2021) are widely used to
evaluate KGQA. However, during dataset con-
struction, only question-answer pairs that yield a
gold SPARQL answer on the reference KG are
retained, while any unanswerable questions are dis-
carded (Yih et al., 2016; Talmor and Berant, 2018b;
Gu et al., 2021). This implicitly assumes that each
question can be answered directly using an exist-
ing fact in the KG, overlooking the reality that
KGs are often incomplete. To address this, recent
works (Xu et al., 2024; Zhou et al., 2025) simulate
KG incompleteness by either randomly deleting
triples from the whole KG or removing those along
the shortest path(s) between the question and an-

swer entities. However, these approaches have a
key limitation: they cannot ensure that sufficient
knowledge remains in the KG to support answer-
ing each question. This can lead to misleading
evaluations, as performance drops may stem from
unanswerable questions rather than limitations in
the model’s reasoning ability.
2.4 LLM Reasoning Evaluation
Evaluation of LLM reasoning typically covers both
logical question answering and logical consistency,
with benchmarks spanning deduction, multiple-
choice, proof generation, and consistency con-
straints such as entailment, negation, and transi-
tivity. Datasets like LogicBench (Parmar et al.,
2024), ProofWriter (Tafjord et al., 2020), and Log-
icNLI (Tian et al., 2021) are widely used for these
tasks, but results indicate that LLMs still struggle
with robust logical reasoning (Cheng et al., 2025;
He et al., 2025).
3 Benchmark Construction
This section presents a general method for con-
structing benchmarks, evaluating KG-RAG meth-
ods that can support different settings of knowledge
incompleteness. The key objective is to create nat-
ural language questions whose answers are not di-
rectly stated in the KG but can be logically inferred
through reasoning over alternative paths.
To achieve this, we first mine high-confidence
logical rules from the KG to identify triples that
are inferable via reasoning. We then remove a sub-
set of these triples while preserving the supporting
facts required for inference. Natural language ques-
tions are generated based on the removed triples,
meaning that models must rely on reasoning rather
than direct retrieval to answer the questions.
3.1 Rule Mining
To ensure that questions in our benchmark require
reasoning rather than direct lookup, we first iden-
tify triples that are logically inferable from other
facts. We achieve this by mining high-confidence
Horn rules from the original KG using the AMIE3
algorithm (Lajus et al., 2020).
AMIE3 is a widely used rule mining system de-
signed to operate efficiently over large-scale KGs.
A logical rule discovered by AMIE3 has the fol-
lowing form ( Horn rules (Horn, 1951)):
B1∧B2∧ ··· ∧ Bn⇒H ,where each item is called an atom , a binary relation
of the form r(X, Y ), in which ris a predicate and
X, Y are variables. The left-hand side of the rule
is a conjunction of body atoms , denoted as B=
B1∧ ··· ∧ Bn, and the right-hand side is the head
atom H. Intuitively, a rule expresses that if the
bodyBholds, then the head His likely to hold as
well.
Asubstitution σmaps every variable occurring
in an atom to a entity that exists in G. For example
applying σ={X7→Justin , Y7→Jaxon}to the
atom hasSibling(X,Y) yields the grounded fact
hasSibling(Justin,Jaxon) . Agrounding of a
ruleB⇒His
σ(B1)∧. . . σ(Bn)⇒σ(H).
Quality Measure. AMIE3 uses the following
metrics to measure the quality of a rule:
•Support. The support of a rule is defined
as the number its groundings for which all
grounded facts are observed in the KG:
support (B⇒H) =
|{σ(H)| ∀i, σ(Bi)∈ G ∧ σ(H)∈ G}| .
•Head coverage. Head coverage (hc) mea-
sures the proportion of observed head ground-
ings in the KG that are successfully explained
by the rule. It is defined as the ratio of the
rule’s support to the number of head ground-
ings in the KG:
hc(B⇒H) =support (B⇒H)
|{σ|σ(H)∈ G}|.
•Confidence. Confidence measures the propor-
tion of body groundings that also lead to the
head being observed in the KG. It is defined
as the ratio of the rule’s support to the number
body groundings in the KG:
confidence (B⇒H) =support (B⇒H)
|{σ|σ(B)∈ G}|.
We retain only rules with high confidence and
sufficient support, filtering out noisy or spurious
patterns. Specifically, we run AMIE3 with a confi-
dence threshold of 0.3, a head coverage threshold
of 0.1, and a maximum rule length of 4. AMIE3
incrementally generates candidate rules via breadth-
first refinement (Lajus et al., 2020) and evaluates
them using confidence and head coverage; only
those meeting the specified thresholds are retained.
Additional details on the rule generation and filter-
ing process are provided in Appendix A.

3.2 Dataset Generation
We aim to generate questions that cannot be an-
swered using direct evidence, but for which suffi-
cient information is implicitly available in the KG.
The core idea is to first remove triples that can be
reliably inferred using high-confidence rules mined
byAMIE3 , and then generate questions based on
these removed triples.
Triple Removal. For each mined rule B⇒H,
we extract up to 30 groundings σ(B⇒H)such
that both the grounded body σ(B)and grounded
headσ(H)exist in the KG. For each such ground-
ing, we remove the head triple σ(H)from the KG
while preserving all body triples σ(B). To ensure
that the removed triples remainlogically inferable
from the remaining KG, we enforce the following
two constraints:
•All grounded body triples required to infer a
removed head must remain in the KG.
•Removing a head triple must not eliminate any
body triple used by other selected groundings.
This guarantees that every removed triple can be in-
ferred through at least one reasoning path provided
by a mined rule.
Question Generation. For each removed triple
r(eh, et), we use GPT-4 to generate a natural-
language question that asks for the answer entity
based on the predicate and a specified topic entity.
To promote diversity, we randomly designate either
the head ehor the tail etas the topic entity, with
the other serving as the answer entity. The exact
prompt template is provided in Appendix D.
Dataset Balancing. KGs typically exhibit a
"long-tail" distribution, where a small number of
entities participate in a disproportionately large
number of triples, while the majority appear only
infrequently (Mohamed et al., 2020; Chen et al.,
2023). This imbalance can cause many generated
questions to share the same answer entity, leading
to biased evaluation.
To reduce answer distribution bias, we apply
frequency-based downsampling to the generated
questions Q, yielding a more balanced subset Q′⊆
Q. As described in Algorithm 1, for each answer
entity a, we retain at most τ· |Q| questions if a
exceeds the frequency threshold τ; otherwise, all
associated questions are kept.Algorithm 1 Downsampling Procedure
Require: Question set Q; threshold τ∈(0,1]
Ensure: Balanced subset Q′⊆ Q
1:LetA ← set of unique answer entities in Q
2:Q′← ∅
3:for all a∈ A do
4:Qa← {q∈ Q | answer (q) =a}
5: if|Qa|> τ· |Q| then
6: Randomly sample Sa⊂ Q a
7: of size ⌊τ· |Q|⌋
8: else
9: Sa← Q a
10: Q′← Q′∪ Sa
11:return Q′
Answer Set Completion. Although each ques-
tion is initially generated based on a single deleted
triple, there may exist multiple correct answers in
the KG. For example, the question “ Who is the
brother of Justin Bieber? ” may have several valid
answers beyond the one used to generate the ques-
tion (e.g., Jaxon Bieber ). To ensure rigorous and
unbiased evaluation, we construct for each ques-
tion a complete set of correct answers using the full
KG before any triple deletions. Specifically, for a
given topic entity and predicate, we identify all tail
entities such that the triple (topic, predicate,
tail) exists in the KG. All such entities are col-
lected as the answer set for that question.
3.3 Dataset Overview
Knowledge Graphs. To support a systematic
evaluation of reasoning under knowledge incom-
pleteness, we construct benchmark datasets based
on two well established KGs: Family (Sadeghian
et al., 2019) and FB15k-237 (Toutanova and Chen,
2015). These datasets differ in size, structure, and
domain coverage, enabling evaluation across both
synthetic and real-world settings.
•Family dataset is a synthetic KG encoding
well-defined familial relationships, such as
father ,mother ,uncle , and aunt . It is con-
structed from multiple family units with log-
ically consistent relation patterns and inter-
pretable schema.
•FB15k-237 is a widely used benchmark de-
rived from Freebase (Dong et al., 2014). The
graph spans 14,541 entities and 237 predi-
cates, covering real-world domains such as
people, locations, and organizations.

Mined Rules. Table 1 summarizes the number
of mined rules for each dataset, categorized by
rule type. The listed types (e.g., symmetry, inver-
sion, composition) correspond to common logical
patterns, while the other category includes more
complex or irregular patterns (See Appendix G for
details).
Rule Type Family FB15k-237
Symmetry: r(x, y)⇒r(y, x) 0 27
Inversion: r1(x, y)⇒r2(y, x) 6 50
Hierarchy: r1(x, y)⇒r2(x, y) 0 76
Composition: r1(x, y)∧r2(y, z)⇒r3(x, z) 56 343
Other 83 570
Total 145 1,066
Table 1: Statistics of mined rules.
Datasets. Each dataset instance consists of (1)
a natural-language question, (2) a topic entity ref-
erenced in the question, and (3) a complete set of
correct answer entities derived from the original
KG. Table 2 presents representative examples from
each dataset. The final question set is randomly
partitioned into training, validation, and test sets
using an 8:1:1 ratio. This split is applied uniformly
across both datasets to ensure consistency.
we provide two retrieval sources per dataset:
•Complete KG : the original KG containing all
triples.
•Incomplete KG : a modified version where se-
lected triples, deemed logically inferable via
AMIE3 -mined rules, are removed (cf. Sec-
tion 3.2).
Table 3 summarizes the number of KG triples
and generated questions in each split for both
datasets, under complete and incomplete KG set-
tings.
4 Evaluation Protocol
4.1 Evaluation Setup
Given a natural-language question q∈ Q, access
to a KG, and a topic entity, the model is tasked with
returning a set of predicted answer entities Pq.
Since KG-RAG models typically produce raw
text sequences as output, we extract the final pre-
diction set Pqby applying string partitioning and
normalizing, following Luo et al. (2024). Details of
this postprocessing step are provided in Appendix
E. Without specific justification, all entities are
represented by randomly assigned indices withouttextual labels (e.g., “Barack Obama” becomes 39)
to ensure that models rely solely on knowledge
from the KG rather than memorized surface forms.
4.2 Evaluation Metrics
Given a set of test questions Q, we denote the
predicted answer set and the gold answer set for
each question q∈ Q asPqandAq, respectively.
The evaluation metrics are defined as follows:
Hits@ Any. Hits@ Any measures the proportion
of questions for which the predicted answer set
overlaps with the gold answer set, i.e., at least one
correct answer is predicted:
Hits@ Any =1
|Q|X
q∈Q1[Pq∩ Aq̸=∅].
Precision andRecall. Precision measures the frac-
tion of predicted answers that are correct, while
recall measures the fraction of gold answers that
are predicted:
Precision =1
|Q|X
q∈Q|Pq∩ Aq|
|Pq|,
Recall =1
|Q|X
q∈Q|Pq∩ Aq|
|Aq|.
F1-score. The F1-score is the harmonic mean of
precision and recall, computed per question and
averaged across all questions:
F1 =1
|Q|X
q∈Q2· |Pq∩ Aq|
|Pq|+|Aq|.
Hits@Hard. We define the hard answer for each
question, denoted as aq∈ A q, as the specific an-
swer entity selected during the question generation
process, i.e., the one whose supporting triple was
intentionally removed from the KG. Hits@Hard
measures the proportion of predictions including
the hard answer. It is defined as:
Hits@Hard =1
|Q|X
q∈Q1[aq∈ Pq].
Hard Hits Rate. We define the Hard Hits Rate
(HHR) as the fraction of correctly answered ques-
tions (i.e., Hits@Any) that include the hard answer
in predictions:
HHR =Hits @Hard
Hits @Any.

Dataset Example
Family Question: Who is 139’s brother? Topic Entity: 139
—
Answer: [205, 138, 2973, 2974]
Direct Evidence: brotherOf(139,205)
Alternative Paths: fatherOf(139,14) ∧uncleOf(205,14) ⇒brotherOf(139,205)
FB15k-237 Question: What is the currency of the estimated budget for 5297 (Annie Hall)? Topic Entity: 5297 (Annie Hall)
—
Answer: [1109 (United States Dollar)]
Direct Evidence: filmEstimatedBudgetCurrency(5297, 1109)
Alternative Paths: filmCountry(5297 (Annie Hall), 2896 (United States of America))
∧locationContains(2896 (United States of America), 9397 (New York))
∧statisticalRegionGdpNominalCurrency(9397 (New York), 1109 (United States Dollar))
⇒filmEstimatedBudgetCurrency(5297 (Annie Hall), 1109 (United States Dollar))
Table 2: Examples from our benchmark datasets. Each instance includes a natural-language question , atopic entity
(provided to the KG-RAG model), and the full set of correct answers . The red-highlighted answer denotes the hard
answer —i.e., the one whose supporting triple has been removed in the incomplete KG setting. We also show the
corresponding direct evidence (the deleted triple) and an alternative path derived from a mined rule that enables
inference of the answer.
Dataset #Triples Train Val Test Total Qs
Family-Complete 17,615 1,749 218 198 2,165
Family-Incomplete 15,785 1,749 218 198 2,165
FB15k-237-Complete 204,087 4,374 535 540 5,449
FB15k-237-Incomplete 198,183 4,374 535 540 5,449
Table 3: Dataset statistics.
5 Experiments and Results
5.1 Overall Performance
We evaluate six representative KG-RAG meth-
ods—RoG (Luo et al., 2024), G-Retriever (He
et al., 2024), GNN-RAG (Mavromatis and Karypis,
2024), PoG (Chen et al., 2024), StructGPT (Jiang
et al., 2023), and ToG (Sun et al., 2023)—on the
benchmark datasets introduced in Section 3.
Figure 1 report Hits@Any and F1-scores across
all methods and datasets. In most cases, both met-
rics drop noticeably when moving from the com-
plete to the incomplete KG setting , highlighting
the challenge posed by missing direct evidence.
Precision and recall follow a similar trend (see Ap-
pendix C).
G-Retriever presents a unique pattern: it shows
similarly low performance across both complete
and incomplete KGs. This is because its retrieval
is based on textual similarity, which often retrieves
both topic and answer entities regardless of KG
completeness. The GNN encoder can partially re-
cover missing links via neighborhood aggregation,
making it less sensitive to missing triples. How-
ever, the lack of explicit reasoning and noisy k-NN
retrieval lead to irrelevant candidates and overalllow F1.
5.2 Impact of Removing Direct Evidence
To examine the impact of removing direct evidence,
we report HHR in Figure 2. Ideally, models capable
of reasoning over alternative paths should maintain
a similar HHR across both complete and incom-
plete KG settings. However, across all models and
datasets, we observe a significant drop in HHR
when moving from the complete to the incom-
plete KG setting . This highlights the limited rea-
soning capabilities of current KG-RAG methods:
while they perform well with direct evidence, their
effectiveness declines sharply when they need to
retrieve alternative paths and reason over it to infer
the answer. Notably, even on the Family dataset,
where relation patterns are simple and should be
easily recognized by LLMs, models exhibit a sub-
stantial decline in recovering the correct answer via
alternative paths when direct evidence is absent.
Training-based methods (e.g., RoG and GNN-
RAG) show a smaller drop in HHR compared
to non-trained methods (e.g., PoG and ToG) ,
suggesting that exposure to incomplete KGs dur-
ing training helps models generalize over indirect
reasoning paths. In contrast, non-trained methods
perform well when direct evidence is available but
struggle significantly when such evidence is miss-
ing, revealing a stronger sensitivity to incomplete-
ness and more limited reasoning capabilities.

Figure 1: Performance comparison of KG-RAG models under incomplete (blue) and complete (red) KG settings,
measured by Hits@Any (top) and F1-Score (bottom).
Figure 2: HHR under different KG settings. (a) Family.
(b) FB15K-237.
5.3 Fine-Grained Analysis by Rule Type
To enable a deeper understanding of model reason-
ing, we conduct a fine-grained analysis of HRR
across different rule types on FB15k-237, focusing
on two representative models: RoG and PoG (Fig-
ure 3). Overall, RoG exhibits greater robustness
to KG incompleteness than PoG across most rule
types, indicating that training-based methods can
better generalize to multi-hop reasoning when di-
rect evidence is removed. An exception arises in
symmetric patterns, where PoG outperforms RoG
under the incomplete setting. This is notable be-
cause symmetric relations (e.g., sibling ) are in-
herently more robust to directionality. If a model
retrieves sibling(Bieber, Jaxon) , it is straight-
forward for an LLM to infer sibling(Jaxon,
Bieber) . RoG’s lower robustness in this seem-
ingly trivial case suggests that training-based meth-
ods may overfit to relational patterns seen during
training, at the cost of generalizing over simpler
relations.
Figure 3: HHR across rule types on FB15k-237 for
(a) RoG and (b) PoG, comparing performance under
complete and incomplete KG settings.
5.4 Influence of Entity Labeling
To assess how different entity labeling schemes
affect KG-RAG performance, we evaluate three
settings for representing entities in the input: (1)
Private ID —randomly assigned IDs with no se-
mantic content, (2) Entity ID —official Freebase
IDs (e.g., /m/02mjmr forBarack Obama ), and (3)
Text Label —natural language labels of entities.
Figure 4 shows the F1-scores of all models under
each representation, for both incomplete (top) and
complete (bottom) KG settings. We observe two
key trends: First, text labels significantly boost
performance. Models consistently achieve higher
scores when entity labels are expressed in natu-
ral language. This suggests that LLMs can effec-
tively leverage their internal knowledge when text
labels are provided. Second, entity IDs provide
limited benefit over random IDs. Surprisingly,
using official entity IDs like /m/02mjmr results in
performance nearly identical to that of randomly
assigned private IDs. This indicates that, despite
LLMs potentially memorizing mappings between

Figure 4: F1-Score comparison of models using Private
ID, Entity ID, and Text Label representations on FB15K-
237 benchmark. (a) Incomplete KG. (b) Complete KG.
surface forms and IDs, they are unable to reason
with these identifiers in generation tasks. Instead,
they appear to treat IDs as opaque tokens unless
the text label is explicit.
5.5 Case Study
To gain deeper insight into the limitations of cur-
rent KG-RAG methods, we conduct a qualitative
analysis of representative failure cases from our
benchmark. Table 4 presents two illustrative exam-
ples, each highlighting a distinct failure pattern: (1)
failure to retrieve relevant reasoning paths, and (2)
incorrect answer generation despite retrieving the
correct context.
(1) Example 1: Retrieval Failure. The
question requires reasoning over the relations
contains(New York City, Manhattan) and
source(Manhattan, HUD) to infer the answer
source(New York City, HUD) . However, the
retriever fails to locate any of these relevant facts.
Instead, it returns spurious paths involving enti-
ties like Pace University , based on weak co-
occurrence signals. As a result, the generator hal-
lucinates an incorrect answer. This type of error
is especially common among non-trained retrieval
methods, which are more sensitive to missing links
and lack robust multi-hop retrieval capabilities.
(2) Example 2: Reasoning Failure. The
retriever returns the correct supporting triple
spouse(Ian Holm, Penelope Wilton) , which
should allow the generator to infer the reverse di-
rection. Nonetheless, the model produces the incor-
rect answer Marriage , likely due to interference
from unrelated paths involving typeOfUnion re-
lations. This suggests that even with accurate re-
trieval, models can fail to distinguish relevant from
irrelevant context during generation.6 Discussion and Conclusion
This work introduces a general methodology for
constructing benchmarks aimed at evaluating the
reasoning capabilities of KG-RAG systems under
conditions of knowledge incompleteness—a real-
istic yet often overlooked challenge in real-world
KGs. Our benchmark construction pipeline explic-
itly removes direct supporting facts while ensuring
that answers remain inferable via alternative paths,
allowing us to isolate and measure true reasoning
performance.
Our empirical study on datasets derived from
Family and FB15k-237 reveals several key limi-
tations of current KG-RAG methods. Most no-
tably, existing KG-RAG models struggle to recover
answers when direct evidence is missing, indicat-
ing limited robustness to incomplete knowledge.
While training-based methods exhibit greater re-
silience, our fine-grained analysis reveals potential
overfitting to specific relation patterns, sometimes
at the expense of generalizing over trivial struc-
tures. Furthermore, we find that textual entity la-
bels substantially improve performance, suggest-
ing that models rely more on retrieving memorized
knowledge than performing symbolic reasoning
over structured data.
These findings point to several promising direc-
tions for future research. First, there is a need
for more advanced retrieval strategies that can ef-
fectively identify alternative paths when direct ev-
idence is missing. Second, reasoning modules
should be designed with greater generalization ca-
pability to avoid overfitting to specific relation pat-
terns. Third, fine-tuning strategies should be care-
fully crafted to enhance retrieval without compro-
mising the LLM’s inherent reasoning ability.
7 Limitation
One limitation of our benchmark lies in the expres-
siveness of the logical rules used for constructing
reasoning questions. Specifically, we rely on rules
mined by AMIE3 , which are restricted to Horn-
style rules. While this rule format is efficient and
widely adopted, it captures only a subset of the
reasoning patterns present in real-world KGs.
As a result, our benchmark primarily evaluates
reasoning over relatively simple logical structures
(e.g., symmetry, transitivity, composition). More
sophisticated forms of inference—such as disjunc-
tive reasoning, aggregation (e.g., counting), and nu-
merical constraints—are not currently represented.

Example 1: Question: What is the source of the estimated number of mortgages for New York City ?Answer: [United States Department of Housing and Urban Development]
Alternative Path: contains(New York City, Manhattan) ∧source(Manhattan, HUD) ⇒source(New York City, HUD)
—
Prediction: [Pace University]
Retrieved Paths: organizationExtra(New York City, Phone Number) ∧serviceLocation(Phone Number, Pace University)
(no path involving mortgage source)
Example 2: Question: Who is Ian Holm ’s spouse? Answer: [Penelope Wilton]
Alternative Paths: spouse(Ian Holm, Penelope Wilton) ⇒spouse(Penelope Wilton, Ian Holm)
—
Prediction: [Marriage]
Retrieved Paths: spouse(Ian Holm, Penelope Wilton)
awardNominee(Ian Holm, Cate Blanchett) ∧typeOfUnion(Cate Blanchett, Marriage)
awardNominee(Ian Holm, Kate Beckinsale) ∧typeOfUnion(Kate Beckinsale, Domestic Partnership)
... (many additional unrelated paths via award nominees)
Table 4: Case study examples from our benchmark illustrating typical failure modes of KG-RAG models. In both
examples, the bold text highlights the topic entity, the green text denotes the expected alternative path, and the red
text marks incorrect predictions.
These reasoning types are important in many real-
world applications and pose greater challenges for
KG-RAG systems.
In future work, it would be valuable to explore
methods for incorporating richer rule types into the
benchmark, either by extending the rule mining
process (e.g., with neural-symbolic rule learners or
probabilistic rule discovery techniques) or by man-
ually curating complex inference templates. This
would allow for a more comprehensive evaluation
of KG-RAG models’ reasoning capabilities beyond
basic logical chaining.
8 Acknowledgements
The authors thank the International Max Planck
Research School for Intelligent Systems (IMPRS-
IS) for supporting Yuqicheng Zhu and Hongkuan
Zhou. The work was partially supported by EU
Projects Graph Massivizer (GA 101093202), en-
RichMyData (GA 101070284), SMARTY (GA
101140087), and the EPSRC project OntoEm
(EP/Y017706/1).
References
Jinheon Baek, Alham Fikri Aji, Jens Lehmann, and
Sung Ju Hwang. 2023. Direct fact retrieval from
knowledge graphs without entity linking. arXiv
preprint arXiv:2305.12416 .
Russa Biswas, Lucie-Aimée Kaffee, Michael Cochez,
Stefania Dumbrava, Theis E Jendal, Matteo Lissan-
drini, Vanessa Lopez, Eneldo Loza Mencía, Heiko
Paulheim, Harald Sack, and 1 others. 2023. Knowl-
edge graph embeddings: open challenges and oppor-
tunities. Transactions on Graph Data and Knowl-
edge , 1(1):4–1.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, BogdanDamoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, Tom Hennigan,
Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, and 9 others. 2022. Improving language
models by retrieving from trillions of tokens. In
ICML , volume 162 of Proceedings of Machine Learn-
ing Research , pages 2206–2240. PMLR.
Lihu Chen, Simon Razniewski, and Gerhard Weikum.
2023. Knowledge base completion for long-tail
entities. In Proceedings of the First Workshop on
Matching From Unstructured and Structured Data
(MATCHING 2023) , pages 99–108.
Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun,
Jieping Ye, and Hui Xiong. 2024. Plan-on-graph:
Self-correcting adaptive planning of large language
model on knowledge graphs. In Proceedings of the
38th Conference on Neural Information Processing
Systems .
Fengxiang Cheng, Haoxuan Li, Fenrong Liu, Robert van
Rooij, Kun Zhang, and Zhouchen Lin. 2025. Empow-
ering llms with logical reasoning: A comprehensive
survey. arXiv preprint arXiv:2502.15652 .
Xin Luna Dong, Evgeniy Gabrilovich, Geremy Heitz,
Wiko Horn, Kevin Murphy, Shaohua Sun, and Wei
Zhang. 2014. From data fusion to knowledge fusion.
Proceedings of the VLDB Endowment , 7(10).
Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald.
2024. Reano: Optimising retrieval-augmented reader
models through knowledge graph generation. In Pro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 2094–2112.
Luis Galárraga, Christina Teflioudi, Katja Hose, and
Fabian M Suchanek. 2015. Fast rule mining in on-
tological knowledge bases with amie+. The VLDB
Journal , 24(6):707–730.
Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy
Liang, Xifeng Yan, and Yu Su. 2021. Beyond iid:
three levels of generalization for question answering
on knowledge bases. In Proceedings of the Web
Conference 2021 , pages 3477–3488. ACM.

Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan
Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
Rossi, Subhabrata Mukherjee, Xianfeng Tang, and 1
others. 2024. Retrieval-augmented generation with
graphs (graphrag). arXiv preprint arXiv:2501.00309 .
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering. Advances in Neural Information
Processing Systems , 37:132876–132907.
Yuan He, Bailan He, Zifeng Ding, Alisia Lupidi,
Yuqicheng Zhu, Shuo Chen, Caiqi Zhang, Jiaoyan
Chen, Yunpu Ma, V olker Tresp, and 1 others. 2025.
Supposedly equivalent facts that aren’t? entity fre-
quency in pre-training induces asymmetry in llms.
arXiv preprint arXiv:2503.22362 .
Alfred Horn. 1951. On sentences which are true of
direct unions of algebras1. The Journal of Symbolic
Logic , 16(1):14–21.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open
domain question answering. In EACL , pages 874–
880. Association for Computational Linguistics.
Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye,
Wayne Xin Zhao, and Ji-Rong Wen. 2023. Struct-
gpt: A general framework for large language model
to reason over structured data. arXiv preprint
arXiv:2305.09645 .
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2020. Generalization
through memorization: Nearest neighbor language
models. In ICLR . OpenReview.net.
Jonathan Lajus, Luis Galárraga, and Fabian Suchanek.
2020. Fast and exact rule mining with amie 3. In
The Semantic Web: 17th International Conference,
ESWC 2020, Heraklion, Crete, Greece, May 31–June
4, 2020, Proceedings 17 , pages 36–52. Springer.
Yunshi Lan, Gaole He, Jinhao Jiang, Jing Jiang,
Wayne Xin Zhao, and Ji-Rong Wen. 2021. A sur-
vey on complex knowledge base question answering:
Methods, challenges and solutions. In Proceedings
of the Thirtieth International Joint Conference on Ar-
tificial Intelligence , pages 4483–4491. International
Joint Conferences on Artificial Intelligence Organi-
zation.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and
Shirui Pan. 2024. Reasoning on graphs: Faithful
and interpretable large language model reasoning. In
ICLR . OpenReview.net.
Costas Mavromatis and George Karypis. 2024. Gnn-
rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139 .Aisha Mohamed, Shameem Parambath, Zoi Kaoudi, and
Ashraf Aboulnaga. 2020. Popularity agnostic evalua-
tion of knowledge graph embeddings. In Conference
on Uncertainty in Artificial Intelligence , pages 1059–
1068. PMLR.
OpenAI. 2024. Chatgpt(3.5)[large language model].
https://chat.openai.com .
Mihir Parmar, Nisarg Patel, Neeraj Varshney, Mutsumi
Nakamura, Man Luo, Santosh Mashetty, Arindam
Mitra, and Chitta Baral. 2024. Logicbench: To-
wards systematic evaluation of logical reasoning
ability of large language models. arXiv preprint
arXiv:2404.15522 .
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey. arXiv preprint arXiv:2408.08921 .
Nico Potyka, Yuqicheng Zhu, Yunjie He, Evgeny Khar-
lamov, and Steffen Staab. 2024. Robust knowledge
extraction from large language models using social
choice theory. In Proceedings of the 23rd Interna-
tional Conference on Autonomous Agents and Multi-
agent Systems , pages 1593–1601.
Yunqi Qiu, Yuanzhuo Wang, Xiaolong Jin, and Kun
Zhang. 2020. Stepwise reasoning for multi-relation
question answering over knowledge graph with weak
supervision. In Proceedings of the 13th international
conference on web search and data mining , pages
474–482.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Trans. Assoc. Comput. Linguistics ,
11:1316–1331.
Ali Sadeghian, Mohammadreza Armandpour, Patrick
Ding, and Daisy Zhe Wang. 2019. Drum: End-to-
end differentiable rule mining on knowledge graphs.
Advances in neural information processing systems ,
32.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel M Ni, Heung-
Yeung Shum, and Jian Guo. 2023. Think-on-
graph: Deep and responsible reasoning of large lan-
guage model on knowledge graph. arXiv preprint
arXiv:2307.07697 .
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. In ICLR . OpenRe-
view.net.
Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter
Clark. 2020. Proofwriter: Generating implications,
proofs, and abductive statements over natural lan-
guage. arXiv preprint arXiv:2012.13048 .

Alon Talmor and Jonathan Berant. 2018a. The web
as a knowledge-base for answering complex ques-
tions. In NAACL-HLT , pages 641–651. Association
for Computational Linguistics.
Alon Talmor and Jonathan Berant. 2018b. The web as
a knowledge-base for answering complex questions.
arXiv preprint arXiv:1803.06643 .
Jidong Tian, Yitian Li, Wenqing Chen, Liqiang Xiao,
Hao He, and Yaohui Jin. 2021. Diagnosing the
first-order logical reasoning ability through logicnli.
InProceedings of the 2021 Conference on Empiri-
cal Methods in Natural Language Processing , pages
3738–3747.
Kristina Toutanova and Danqi Chen. 2015. Observed
versus latent features for knowledge base and text
inference. In Proceedings of the 3rd workshop on
continuous vector space models and their composi-
tionality , pages 57–66.
Yao Xu, Shizhu He, Jiabei Chen, Zihao Wang, Yangqiu
Song, Hanghang Tong, Guang Liu, Kang Liu,
and Jun Zhao. 2024. Generate-on-graph: Treat
llm as both agent and kg in incomplete knowl-
edge graph question answering. arXiv preprint
arXiv:2404.14741 .
Liang Yao, Chengsheng Mao, and Yuan Luo. 2019. Kg-
bert: Bert for knowledge graph completion. arXiv
preprint arXiv:1909.03193 .
Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou,
and Caiming Xiong. 2021. Rng-kbqa: Generation
augmented iterative ranking for knowledge base ques-
tion answering. arXiv preprint arXiv:2109.08678 .
Wen-tau Yih, Matthew Richardson, Christopher Meek,
Ming-Wei Chang, and Jina Suh. 2016. The value of
semantic parse labeling for knowledge base question
answering. In ACL (2) . The Association for Com-
puter Linguistics.
Dongzhuoran Zhou, Yuqicheng Zhu, Yuan He, Jiaoyan
Chen, Evgeny Kharlamov, and Steffen Staab. 2025.
Evaluating knowledge graph based retrieval aug-
mented generation methods under knowledge incom-
pleteness. arXiv preprint arXiv:2504.05163 .
Yuqicheng Zhu, Daniel Hernández, Yuan He, Zifeng
Ding, Bo Xiong, Evgeny Kharlamov, and Steffen
Staab. 2025a. Predicate-conditional conformalized
answer sets for knowledge graph embeddings. In
Findings of the Association for Computational Lin-
guistics: ACL 2025 , pages 4145–4167, Vienna, Aus-
tria. Association for Computational Linguistics.
Yuqicheng Zhu, Nico Potyka, Mojtaba Nayyeri,
Bo Xiong, Yunjie He, Evgeny Kharlamov, and Stef-
fen Staab. 2024. Predictive multiplicity of knowl-
edge graph embeddings in link prediction. In Find-
ings of the Association for Computational Linguistics:
EMNLP 2024 , pages 334–354, Miami, Florida, USA.
Association for Computational Linguistics.Yuqicheng Zhu, Nico Potyka, Jiarong Pan, Bo Xiong,
Yunjie He, Evgeny Kharlamov, and Steffen Staab.
2025b. Conformalized answer set prediction for
knowledge graph embedding. In Proceedings of
the 2025 Conference of the Nations of the Ameri-
cas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume
1: Long Papers) , pages 731–750.

A Details of Rule Mining
A.1 AMIE3 Candidate Rule Refinement
Refinement is carried out using a set of operators
that generate new candidate rules:
•Dangling atoms, which introduce a new vari-
able connected to an existing one;
•Closing atoms, which connect two existing
variables;
•Instantiated atoms, which introduce a constant
and connect it to an existing variable.
AMIE3 generate candidate rules by a refinement
process using a classical breadth-first search (La-
jus et al., 2020). It begins with rules that contain
only a head atom (e.g. ⇒hasSibling (X, Y ))
and refines them by adding atoms to the body. For
example, it may generate the refined rule:
hasParent (X, Z )∧hasChild (Z, Y)
⇒hasSibling (X, Y ).
This refinement step connects existing variables
and introduces new ones, gradually building mean-
ingful patterns.
A.2 AMIE3 Hyperparameter Settings
We use AMIE3 with a confidence threshold of 0.3
and a PCA confidence threshold θPCAof 0.4 for
both datasets. The maximum rule length is set
to 3 for Family to avoid overly complex patterns,
and 4 for FB15k-237 to allow richer rules. See
Appendix B.1 for the definition of PCA confidence.
B Properties of Horn Rules Mined by
AMIE3
AMIE3 mines logical rules from knowledge graphs
in the form of (Horn) rules:
B1∧B2∧ ··· ∧ Bn=⇒H
where BiandHare atoms of the form r(X, Y ). To
ensure interpretability and practical utility, AMIE3
imposes the following structural properties on all
mined rules:
•Connectedness: All atoms in the rule are
transitively connected via shared variables or
entities. This prevents rules with indepen-
dent, unrelated facts (e.g., diedIn (x, y) =⇒
wasBornIn (w, z)). Two atoms are connectedif they share a variable or entity; a rule is con-
nected if every atom is connected transitively
to every other atom.
•Closedness: Every variable in the rule ap-
pears at least twice (i.e., in at least two atoms).
This avoids rules that merely predict the exis-
tence of some fact without specifying how it
relates to the body, such as diedIn (x, y) =⇒
∃z:wasBornIn (x, z).
•Safety: All variables in the head atom also
appear in at least one body atom. This en-
sures that the rule’s predictions are grounded
by the body atoms and avoids uninstantiated
variables in the conclusion.
These restrictions are widely adopted in KG rule
mining (Galárraga et al., 2015; Lajus et al., 2020) to
guarantee that discovered rules are logically well-
formed and meaningful for downstream reasoning
tasks.
B.1 PCA Confidence
To understand the concept of rule mining better for
the reader we simplified notation of confidence in
main body. Note AMIE3 also supports a more opti-
mistic confidence metric known as PCA confidence ,
which adjusts standard confidence to account for
incompleteness in the KG.
Motivation. Standard confidence for a rule is
defined as the proportion of its correct predictions
among all possible predictions suggested by the
rule. However, this metric is known to be pes-
simistic for knowledge graphs, which are typically
incomplete: many missing triples may be true but
unobserved, unfairly penalizing a rule’s apparent
reliability.
Definition. To address this, AMIE3 introduces
PCA confidence (Partial Completeness Assumption
confidence) (Galárraga et al., 2015), an optimistic
variant that partially compensates for KG incom-
pleteness. Given a rule of the form
B1∧ ··· ∧ Bn=⇒r(x, y)
thestandard confidence is
conf( R) =|{(x, y) :B1∧ ··· ∧ Bn∧r(x, y)}|
|{(x, y) :B1∧ ··· ∧ Bn}|
where the denominator counts all predictions the
rule could possibly make, and the numerator counts
those that are actually present in the KG.

PCA confidence modifies the denominator to
include only those (x, y)pairs for which at least
oner(x, y′)triple is known for the subject x. That
is, the rule is only penalized for predictions about
entities for which we have observed at least some
information about the target relation. Formally,
conf PCA(R) =
|{(x, y) :B1∧ ··· ∧ Bn∧r(x, y)}|
|{(x, y) :B1∧ ··· ∧ Bn∧ ∃y′:r(x, y′)}|
Here, the denominator sums only over those x
for which some y′exists such that r(x, y′)is ob-
served in the KG.
Intuition. This approach assumes that, for any
entity xfor which at least one fact r(x, y′)is
known, the KG is "locally complete" with respect
torforx—so if the rule predicts other r(x, y)facts
forx, and they are missing, we treat them as truly
missing (i.e., as counterexamples to the rule). But
for entities where no r(x, y)fact is observed at all,
the rule is not penalized for predicting additional
facts.
Comparison. PCA confidence thus provides a
more optimistic and fairer assessment of a rule’s
precision in the presence of incomplete data. It
is widely adopted in KG rule mining, and is the
default metric for filtering and ranking rules in
AMIE3.
For further details, see (Galárraga et al., 2015).
B.2 Rule Mining Procedure
Algorithm 2 AMIE3
Require: Knowledge graph G, maximum rule
length l, PCA confidence threshold θPCA, and
head coverage threshold θhc.
Ensure: Set of mined rules R.
1:q←all rules of the form ⊤ ⇒ r(X, Y )
2:R ← ∅
3:while qis not empty do
4: R←q.dequeue()
5: ifSatisfiesRuleCriteria (R)then
6: R ← R ∪ { R}
7: iflen(R)< landθPCA(R)<1.0then
8: for all Rc∈refine (R)do
9: ifhc(Rc)≥θhcandRc/∈qthen
10: q.enqueue( Rc)
11:return R
AMIE3 generate candidate rules by a refinement
process using a classical breadth-first search (Lajuset al., 2020). Algorithm 2 summarizes the rule min-
ing process of AMIE3 . The algorithm starts with
an initial set of rules that contain only a head atom
(i.e.⊤ ⇒ r(X, Y ), where ⊤denotes an empty
body) and maintains a queue of rule candidates
(Line 1). At each step, AMIE3 dequeues a rule R
from the queue and evaluates whether it satisfies
three criteria (Line 5):
•the rule is closed (i.e., all variables in at least
two atoms),
• its PCA confidence is higher than θPCA,
•its PCA confidence is higher than the confi-
dence of all previously mined rules with the
same head atom as Rand a subset of its body
atoms.
If these conditions are met, the rule is added to the
final output set R.
IfRhas fewer than latoms and its confidence
can still be improved (Line 8), AMIE3 applies a
refine operator (Line 9) that generates new can-
didate rules by adding a body atom (details in Ap-
pendix A). Refined rules are added to the queue
only if they have sufficient head coverage (Line
11) and have not already been explored. This pro-
cess continues until the queue is empty, at which
point all high-quality rules satisfying the specified
constraints have been discovered.
B.3 Benchmark Construction Code and Data
We release the source code for benchmark con-
struction, along with the Family and FB15k-
237 benchmark datasets, at https://anonymous.
4open.science/r/INCK-EA16 .
C Additional Results of the experiment
Figure 5 presents the recall and precision of all eval-
uated KG-RAG models on the constructed bench-
marks.
D Prompt Template
Prompt for generating questions from triples:
You are an expert in knowledge graph question
generation.
Given:
Removed Triple: ({entity_h}, {predicate_T}, {
entity_t})
Question Entity: {topic_entity}
Answer Entity: {answer_entity}

Figure 5: Performance comparison of KG-RAG models under incomplete (blue) and complete (red) KG settings,
measured by Recall (top) and Precision (bottom).
Write a clear, natural-language question that
asks for the Answer Entity, using the given
predicate and Topic Entity.
Requirements:
- Express the predicate {predicate_T} naturally
(paraphrasing allowed, but preserve core
meaning; e.g., "wife_of" -> "wife").
- Mention the Topic Entity {topic_entity}.
- The answer should be the Answer Entity {
answer_entity}.
- Do not mention the Answer Entity {
answer_entity} in the question.
- Do not ask a yes/no question.
- Output only the question as plain text.
Example:
Removed Triple: ("Alice", "wife_of", "Carol")
Question Entity: Carol
Answer Entity: Alice
Output:
Who is Carol’s wife?
Now, generate the question for:
Removed Triple: ({entity_h}, {predicate_T}, {
entity_t})
Question Entity: {topic_entity}
Answer Entity: {answer_entity}
To ensure reproducibility and mitigate random-
ness in LLM outputs (Potyka et al., 2024), we set
the generation temperature to 0 in all experiments.
E Detailed Evaluation Settings
All evaluated models are required to produce their
predictions as a structured list of answers , but in
practice, the model output is often a raw string Pstr
(e.g., "Paris, London" or"Paris London" ). To
obtain a set-valued prediction suitable for evalua-
tion, we first apply a splitting function split( Pstr),
which splits the raw string into a list of answer
strings P= [p1, p2, . . . , p n]using delimiters such
as commas, spaces, or newlines as appropriate.We then define a normalization function
norm(·), which converts each answer string to low-
ercase, removes articles ( a,an,the), punctuation,
and extra whitespace, and eliminates the special
token <pad> if present. The final prediction set is
then defined as P={norm( p)|p∈P}, i.e., the
set of unique normalized predictions. The same
normalization is applied to each gold answer in the
listAto obtain the set A.
All evaluation metrics are computed based on
the resulting sets of normalized predictions Pand
gold answers A.
Algorithm 3 Output Processing
Require: Model output string Pstr, gold answer
listA
Ensure: Normalized prediction set P, normalized
gold set A
1:P←split( Pstr)
2:P ← { norm( p)|p∈P}
3:A ← { norm( a)|a∈A}
4:return P,A
F Baseline Details
Unless otherwise specified, for all methods we
use the LLM backbone and hyperparameters as
described in the original papers.
RoG, G-Retriever, and GNN-RAG are each
trained and evaluated separately on the 8:1:1 train-
ing split of each dataset (Family and FB15k-237)
using a single NVIDIA H200 GPU, as described
in Section 3.3. For RoG, we use LLaMA2-Chat-
7B as the LLM backbone, instruction-finetuned on
the training split of Family or FB15K-237 for 3
epochs. The batch size is set to 4, the learning

rate to 2×10−5, and a cosine learning rate sched-
uler with a warmup ratio of 0.03 is adopted (Luo
et al., 2024). For G-Retriever, the GNN back-
bone is a Graph Transformer (4 layers, 4 attention
heads per layer, hidden size 1024) with LLaMA2-
7B as the LLM. Retrieval hyperparameters and
optimization follow He et al. (2024). For GNN-
RAG (Mavromatis and Karypis, 2024), we use
the recommended ReaRev backbone and sBERT
encoder; the GNN component is trained for 200
epochs with 80 epochs of warmup and a patience of
5 for early stopping. All random seeds are fixed for
reproducibility. For PoG (Chen et al., 2024), Struct-
GPT (Jiang et al., 2023), and ToG (Sun et al., 2024),
we use GPT-3.5-turbo as the underlying LLM, and
the original prompt and generation settings from
each method.
G Detailed Analysis of Other Rule Types
TheOther category in Table 1 encompasses a broad
range of logical rules that do not fall into stan-
dard symmetry, inversion, hierarchy, or composi-
tion classes. Below we summarize the main pat-
terns observed, provide representative examples,
and discuss their impact on model performance.
Longer Compositional Chains. Rules involving
three,
r1(x, y)∧r2(y, z)∧r3(z, w)⇒r4(x, w)
Triangle Patterns. Rules connecting three enti-
ties in a triangle motif,
r1(x, y)∧r2(x, z)⇒r3(y, z)
Intersection Rules. Rules where multiple body
atoms share the same argument,
r1(x, y)∧r2(x, y)⇒r3(x, y)
Other Patterns. Some rules do not exhibit sim-
ple interpretable motifs, involving unusual variable
binding or rare predicate combinations. Like recur-
sive rules (check AMIE3 (Lajus et al., 2020) for
more details)
H Personal Identification Issue in
FB15k-237
While FB15k-237 contains information about indi-
viduals, it typically focuses on well-known public
figures such as celebrities, politicians, and histori-
cal figures. Since this information is already widelyavailable online and in various public sources, its
inclusion in Freebase doesn’t significantly com-
promise individual privacy compared to datasets
containing sensitive personal information.
I AI Assistants In Writing
We use ChatGPT (OpenAI, 2024) to enhance our
writing skills, abstaining from its use in research
and coding endeavors.