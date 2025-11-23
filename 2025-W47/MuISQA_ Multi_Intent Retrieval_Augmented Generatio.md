# MuISQA: Multi-Intent Retrieval-Augmented Generation for Scientific Question Answering

**Authors**: Zhiyuan Li, Haisheng Yu, Guangchuan Guo, Nan Zhou, Jiajun Zhang

**Published**: 2025-11-20 12:03:36

**PDF URL**: [https://arxiv.org/pdf/2511.16283v1](https://arxiv.org/pdf/2511.16283v1)

## Abstract
Complex scientific questions often entail multiple intents, such as identifying gene mutations and linking them to related diseases. These tasks require evidence from diverse sources and multi-hop reasoning, while conventional retrieval-augmented generation (RAG) systems are usually single-intent oriented, leading to incomplete evidence coverage. To assess this limitation, we introduce the Multi-Intent Scientific Question Answering (MuISQA) benchmark, which is designed to evaluate RAG systems on heterogeneous evidence coverage across sub-questions. In addition, we propose an intent-aware retrieval framework that leverages large language models (LLMs) to hypothesize potential answers, decompose them into intent-specific queries, and retrieve supporting passages for each underlying intent. The retrieved fragments are then aggregated and re-ranked via Reciprocal Rank Fusion (RRF) to balance coverage across diverse intents while reducing redundancy. Experiments on both MuISQA benchmark and other general RAG datasets demonstrate that our method consistently outperforms conventional approaches, particularly in retrieval accuracy and evidence coverage.

## Full Text


<!-- PDF content starts -->

MuISQA: Multi-Intent Retrieval-Augmented Generation for Scientific
Question Answering
Zhiyuan Li1,2, Haisheng Yu1, Guangchuan Guo1, Nan Zhou1, Jiajun Zhang1,2,3,4∗
1Zhongke Zidong Taichu (Beijing), China
2Institute of Automation, Chinese Academy of Sciences, China
3School of Artificial Intelligence, University of Chinese Academy of Sciences, China
4Wuhan AI Research
{lizhiyuan, yuhaisheng, guoguangchuan, zhounan}@taichu.ai
jjzhang@nlpr.ia.ac.cn
Abstract
Complex scientific questions often en-
tail multiple intents, such as identifying
gene mutations and linking them to re-
lated diseases. These tasks require ev-
idence from diverse sources and multi-
hop reasoning, while conventional retrieval-
augmented generation (RAG) systems are
usually single-intent oriented, leading to in-
complete evidence coverage. To assess this
limitation, we introduce the Multi-Intent
Scientific Question Answering (MuISQA)
benchmark, which is designed to evaluate
RAG systems on heterogeneous evidence
coverage across sub-questions. In addition,
we propose an intent-aware retrieval frame-
work that leverages large language mod-
els (LLMs) to hypothesize potential an-
swers, decompose them into intent-specific
queries, and retrieve supporting passages for
each underlying intent. The retrieved frag-
ments are then aggregated and re-ranked
via Reciprocal Rank Fusion (RRF) to bal-
ance coverage across diverse intents while
reducing redundancy. Experiments on
both MuISQA benchmark and other general
RAG datasets demonstrate that our method
consistently outperforms conventional ap-
proaches, particularly in retrieval accuracy
and evidence coverage.
1 Introduction
Retrieval-augmented generation (RAG) enhances
Large Language Models (LLMs) with access
to external knowledge sources, improving their
factual reliability and cross-domain generaliza-
tion (Lee et al., 2019; Karpukhin et al., 2020; Mao
et al., 2021). At its core lies a retrieval module,
typically implemented using dense retrieval tech-
niques that exploit semantic embeddings to locate
∗Corresponding author
The proposed MuISQA dataset and related codes can be
found at https://github.com/Zhiyuan-Li-John/MuISQA
Multi-Intent RAG
Multi-document RAG Multi-hop RAG
question :
sub-intent1 : sub-intent2 :What are ALB gene mutations and their related diseases? 
Identify the mutation of ALB gene 
... (c.714G>A; p.Trp238*) 
sequesce analysis ...    ... different mutations of  ...causeanalbuminemia
... molecular ... causingcongenital analbuminemia 
... of ALB ... c.725G>A ... 
(p.Arg242*) variant is  ... FDH and albumin gene  mutation analysis for ...Find diseases linked to each mutation 
... to this ... albumin ...
c.412C>T and c.802G>T   
Figure 1: An example from our MuISQA benchmark,
challenging RAG systems with multi-document re-
trieval and multi-hop reasoning.
relevant evidence (Gao et al., 2023a; Wang et al.,
2023). These approaches have achieved remark-
able success on traditional question answering
(QA) benchmarks (Yang et al., 2018; Kwiatkowski
et al., 2019; Ho et al., 2020), particularly with
recent large-context embedding models such as
Qwen3-Embedding-8B (Zhang et al., 2025) or
BGE-M3 (Chen et al., 2024b).
Despite recent progress, most existing
RAG systems (Zhao et al., 2024) and bench-
marks (Singh et al., 2025) are built on a
single-intent assumption, where each question
corresponds to one canonical answer, and retrieval
performance is measured against that unique tar-
get. Widely used datasets such as TriviaQA (Joshi
et al., 2017) and Natural Questions (Kwiatkowski
et al., 2019) further reinforce this paradigm by
annotating a single gold span per query, with
metrics like nDCG (Jeunen et al., 2024) and
Recall@K (Kynkäänniemi et al., 2019) designed
accordingly. However, many scientific questions
naturally entail multiple correlated intents. For
example, a biomedical query about the human
gene ALB may involve several mutations, each
associated with different diseases. As illustrated
in Figure 1, answering such questions requiresarXiv:2511.16283v1  [cs.AI]  20 Nov 2025

aggregating evidence across multiple documents
and performing multi-hop reasoning for compre-
hensive coverage. Existing RAG systems (Yu
et al., 2024; Fan et al., 2024) tend to focus on one
dominant answer, repeatedly retrieving redundant
evidence while overlooking complementary frag-
ments that support alternative perspectives (Xie
et al., 2025).
To systematically investigate this limitation,
we introduce theMulti-IntentScientificQuestion
Answering (MuISQA) benchmark, which is de-
signed to evaluate how RAG systems handle ques-
tions with multiple correlated intents. MuISQA
covers five scientific domains, including biol-
ogy, chemistry, geography, medicine, and physics,
with each question annotated for diverse sub-
intents and their corresponding answers. Unlike
prior benchmarks (Kwiatkowski et al., 2019; Ho
et al., 2020) that primarily emphasize precision
or recall, MuISQA introduces evaluation metrics
across three key dimensions: (i)Query formula-
tion, measuring the ability to capture distinct in-
tents; (ii)Passage retrieval, assessing coverage
over different subtopics; and (iii)Answer gener-
ation, evaluating both accuracy and completeness
of final responses.
Building on these insights, we further develop
an intent-aware retrieval framework that enhances
both retrieval diversity and evidence coverage. It
first leverages LLMs to hypothesize potential an-
swers for the question and decomposes them into
intent-specific queries. Unlike traditional query-
rewriting methods (Ma et al., 2023; Jagerman
et al., 2023) that generate semantically similar
variants, our approach explicitly injects distinct
hypothetical information into each query, broad-
ening the search intent and improving evidence
diversity. Compared with hypothetical document
approaches such as HyDE (Gao et al., 2023a),
which rely on a single synthetic passage to guide
retrieval, our method promotes retrieval diversifi-
cation by decomposing multiple hypotheses into
independent queries, each retrieving relevant pas-
sages for one intent via embedding similarity.
The retrieved chunks are then aggregated and re-
ranked using Reciprocal Rank Fusion (RRF) (Cor-
mack et al., 2009), which balances complementary
evidence while reducing redundancy from single
intent.
We evaluate our framework on both the pro-
posed MuISQA benchmark and several gen-eral RAG datasets, including TriviaQA (Joshi
et al., 2017), HotpotQA (Yang et al., 2018),
NQ (Kwiatkowski et al., 2019), 2WikiMQA (Ho
et al., 2020), and MuSiQue (Trivedi et al., 2022),
to assess its effectiveness and generalization. On
MuISQA, our approach shows substantial gains
in query efficiency and retrieval coverage over
prior methods (Ma et al., 2023; Gao et al., 2023a;
Xie et al., 2025), highlighting its ability to cap-
ture diverse intents and retrieve complementary
evidence. Moreover, it surpasses recent state-
of-the-art (SOTA) RAG systems (Jimenez et al.,
2024; Shen et al., 2025; Zhu et al., 2025) on gen-
eral datasets, particularly on multi-hop reasoning
tasks, demonstrating strong generalization. Inter-
estingly, our analysis also uncovers that imperfect
hypothetical generation, which is often regarded
as LLM hallucination, can occasionally facilitate
retrieval by guiding the search toward target pas-
sages, contrasting with the limitations previously
reported for HyDE (Gao et al., 2023a).
2 Related Work
2.1 RAG Benchmarks
RAG benchmarks (Yu et al., 2024; Fan et al.,
2024) have been central to evaluating systems
across question answering (Talmor et al., 2019;
Chen et al., 2021), document retrieval (Wang et al.,
2024; Wasserman et al., 2025), and summariza-
tion (Wang et al., 2020; Petroni et al., 2021). Early
datasets such as HotpotQA (Yang et al., 2018)
and TriviaQA (Joshi et al., 2017) largely follow
a single-answer paradigm, emphasizing multi-hop
reasoning (Ho et al., 2020; Trivedi et al., 2022) and
intensive knowledge (Kwiatkowski et al., 2019)
respectively. With advances in LLMs (Liu et al.,
2024; Yang et al., 2025; OpenAI, 2025) and em-
bedding techniques (Chen et al., 2024a; Zhang
et al., 2025), performance on these benchmarks
has improved substantially. More recent works ex-
tend evaluation to multimodal (Wasserman et al.,
2025; Xia et al., 2024) and graph-structured re-
trieval tasks (Xiao et al., 2025), reflecting a shift
toward more complex and realistic settings. Re-
cently, a few studies have also begun exploring the
multi-intent regime, where a question may contain
multiple valid answers across distinct subtopics.
For example, Wang et al. (2025) investigates re-
trieval under conflicting or imbalanced evidence
and finds that models tend to favor dominant an-
swers, while Xie et al. (2025) decomposes open-

        Human
 Verification
Evaluation Metrics 
           Design
    LLM-based
Pre-annotation
Topic and Article          Collection
Article Search and Collection
Biology, Chemistry, Geography, Medicine, Physics 
Topic
Evaluation MetricsPre-annotation and Verification
Stage1:  Query formulation
diversity
Stage2:  Passage retrieval
coverageMuISQA Benchmark Construction
Stage3:  Ansewr generation
performanceFigure 2: The structure of MuISQA benchmark construction. It follows synthesis in four core levels: (1) Topic and
article collection, (2) LLM-based pre-annotation, (3) Human verification, and (4) Evaluation metrics design.
ended questions into sub-queries to assess evi-
dence coverage. However, these efforts primar-
ily focus on answer-side evaluation, offering lim-
ited insights into query diversification or retrieval
coverage. In contrast, our benchmark directly tar-
gets the multi-intent setting and introduces metrics
that disentangle performance across query formu-
lation, evidence retrieval, and answer generation,
providing a finer-grained diagnosis of system be-
haviour and clear guidance for improving RAG
systems.
2.2 Query Optimization in RAG
Query optimization has been extensively ex-
plored as an approach to enhance retrieval ac-
curacy in RAG systems (Gao et al., 2023b).
Early studies focused on query expansion and
rewriting techniques, including classical relevance
feedback methods such as Rocchio (ROCCHIO,
1971), concept-based expansion (Qiu and Frei,
1993), and pseudo-relevance feedback (Xu and
Croft, 1996). Recent advances have shifted to-
ward generative approaches, where LLMs syn-
thesize richer or more diverse query representa-
tions. For instance, Query2Doc (Wang et al.,
2023) and HyDE (Gao et al., 2023a) generate
pseudo-documents as expanded queries, DMQR-
RAG produces rewrites with varying specificity
to improve coverage (Li et al., 2024), and RaFe
leverages reranker feedback to train more effec-
tive query rewrites (Mao et al., 2024). Another
recent work (Xie et al., 2025) decomposes the
question into core, background, and follow-up
components to improve both retrieval coverage
and answer generation. While these approaches
share certain similarities and exhibit notable inno-
vations in query optimization, they overlook the
challenge of handling multiple underlying intents.In contrast, our framework hypothesizes diverse
plausible answers, decomposes them into intent-
specific queries, and aggregates retrieved chunks
through RRF algorithm, directly addressing the
multi-intent coverage problem rather than merely
refining precision on a dominant intent.
3 MuISQA Benchmark
To construct MuISQA benchmark, we design a
hybrid pipeline that integrates automated data ac-
quisition with LLM-assisted annotation to effi-
ciently build the dataset. As shown in Figure 2,
the construction process consists of four main
stages: (1)Topic and article collection, where
domain-relevant documents are gathered across
multiple scientific disciplines; (2)LLM-based
pre-annotation, which generates preliminary la-
bels for each article through the LLM; (3)Human
verification and correction, ensuring factual ac-
curacy, coherence, and domain reliability in the fi-
nal dataset; and (4)Evaluation metrics design,
defining measures to assess system performance
across query formulation, evidence retrieval, and
answer generation. Representative examples from
MuISQA are presented in Appendix 8.1.
3.1 Topic and Article Collection
The MuISQA benchmark spans five scientific do-
mains, includingbiology,chemistry,geography,
medicine, andphysics. For each domain, we cu-
rate 100 multi-intent questions that admit multiple
valid answers, resulting in 500 questions overall.
To construct the evidence corpus, representative
keywords are extracted from each question and
used to search on the arXiv website for relevant
open-access articles. For every question, we re-
trieve 20 articles without manual filtering, collect-
ing 10,000 documents in total. This setup mirrors

realistic retrieval conditions, where relevant and
irrelevant documents coexist, capturing the noise
and diversity of real-world scenarios.
3.2 LLM-based Pre-annotation
To enable LLM-assisted pre-annotation, we devel-
oped a dedicated annotation platform, as shown
on the right side of Figure 2. The system loads
each question together with its retrieved docu-
ments, displaying the full article text on the left
panel and the annotation interface on the right.
We integrate the DeepSeek-V3 (Liu et al., 2024)
into the tool and implement a one-click annotation
function (Auto_Fill (LLM)), which automatically
reads the document, combines it with the associ-
ated question, and generates a preliminary answer
waiting for human review. Further implementation
details are provided in Appendix 8.2.
3.3 Human Verification and Correction
To support efficient human verification and cor-
rection, our platform integrates interactive features
that streamline the validation of generated annota-
tions. The left panel provides a keyword search
function that highlights relevant terms within the
document, helping annotators quickly locate evi-
dence linked to the pre-annotated answers. When
inconsistencies or missing details are found, an-
notators can directly edit the answers in the right
interface. Once verification is complete, the re-
sults can be saved, and the annotator can move
to the next document via theSave_and_Nextbut-
ton. Further implementation details are provided
in Appendix 8.3.
3.4 Evaluation Metrics Design
Unlike previous RAG benchmarks that primarily
focus on one canonical answer, each question in
MuISQA involves multiple intents leading to mul-
tiple correct answers, with each answer poten-
tially supported by several semantically similar
passages drawn from different documents. This
structure makes conventional metrics such as Ex-
act Match (EM) or Recall inadequate for capturing
system performance. To address this, we design
tailored evaluation metrics specifically for multi-
intent questions, disentangling model capability
across three critical stages:
• Query formulation, evaluating the model’s
ability to capture diverse underlying intents;• Passage retrieval, measuring the coverage of
retrieved evidence across subtopics;
• Answer generation, assessing both the fac-
tual accuracy and completeness of the final
response.
For query formulation, rather than directly eval-
uating intent diversity, we introduce a metric
called vector entropy to quantify the informational
complexity of query representations. Given a
query consisting ofmsentences with embedding
vectors{v(1), . . . , v(m)}, each vector is first nor-
malized into a probability distribution over dimen-
sions:
p(s)
i=|v(s)
i|
P
j|v(s)
j|,(1)
wherev(s)
irepresents thei-th component of the
vector.p(s)
irepresents the normalized probability
of thes-th sentence on thei-th dimension. We
then compute the vector entropy as:
pmix=1
mmX
s=1p(s),(2)
Hmix=−X
pmixlogp mix,(3)
wherep mixrepresents the overall semantic distri-
bution of the entire query,H mixindicates the di-
versity of embedding semantics.
For passage retrieval, unlike conventional RAG
evaluations that simply compute recall based on
the number of retrieved passages, multi-intent
questions require assessing how well a system re-
trieves diverse and informative content. We there-
fore propose the Information Recall Rate (IRR),
which measures the amount of distinct factual
information recovered from retrieved passages.
Specifically, IRR extracts all heterogeneous fac-
tual units within the retrieved set that align with
the gold answers. Given a query with a set of re-
trieved passagesR={r 1, . . . , r n}and the cor-
responding gold-standard answers represented as
factual unitsG={g 1, . . . , g k}, we first extract
all factual units from the retrieved passages that
align with the gold set, forming the subsetR∗=
R ∩ G. The ratio between the number of correctly
retrieved unique information units and the total
number of gold information units is defined as:
IRR=|R∗|
|G|,(4)

What drugs can treat Alzheimer's disease and what  
are their activate molecule and side effects?  
LLM-only
Inference
Donepezil  is a medication ... Alzheimer's Disease. The
active compound is Donepezil hydrochloride , which ... 
The common adverse effects include  nausea , vomiting,  
diarrhea , insomnia , muscle  cramps , and fatigue .
Rivastigmine  is another drug ... Alzheimer's Disease ... 
active compound is Rivastigmine tartrate  ... adverse 
effects include nausea , vomiting , weight loss , dizziness , 
headache , and abdominal pain .
Galatamine  is prescribed for ... Alzheimer's Disease ... 
Its Galatamine hydrobromide  ... active compound ... 
Common adverse effects are nausea ,vomiting , diarrhea , 
dizziness , headache , and decreased appetite . 
Question:
Hypothesize 
potential answers:
Element	
Splitting
1st:  Donepezil  is a drug used for Alzheimer's 
        Disease.
4th:  Rivastigmine  is a drug used for Alzheimer's
         Disease.
5th:  The active molecule in Rivastigmine is
            Rivastigmine tartrate .
nth:  ...... 3rd:  Adverse effects of Donepezil include
         nausea , vomiting , diarrhea , insomnia , 
         muscle cramps , and fatigue . 2nd:  The active molecule in Donepezil is   
         Donepezil hydrochloride .
Alzheimer's 
  Disease-1
Alzheimer's 
  Disease-2
Alzheimer's 
  Disease-3Collected Articles: Chunks:
Splitted Queries:... ...
...
Alzheimer's 
  Disease-20
Document
Chunking
Splitted 
Chunk-1
Splitted 
Chunk-2
Splitted 
Chunk-M
1st QuerySplitted Chunk-15
Splitted Chunk-28
Splitted Chunk-43
...
2nd Query
nth Query
retrieve
retrieve
retrieve
Splitted Chunk-9
Splitted Chunk-56
Splitted Chunk-15
...
...
3rd Query
retrieveSplitted Chunk-71
Splitted Chunk-88
Splitted Chunk-33
...
Splitted Chunk-134
Splitted Chunk-217
Splitted Chunk-266
... ...Retrieve:Splitted Chunk-15
Splitted Chunk-9
Splitted Chunk-71
Splitted Chunk-134
...Splitted Chunk-28
Splitted Chunk-56
Splitted Chunk-88
Splitted Chunk-43
Splitted Chunk-33
Splitted Chunk-266Splitted Chunk-217
...
RRF Rerank:Figure 3: The overview of our proposed intent-aware retrieval framework. The LLM first generates hypothetical
answers and decomposes them into diverse intent-specific queries. These queries are then used to retrieve relevant
document chunks, which are re-ranked using the RRF algorithm to ensure comprehensive coverage of evidence.
where|R∗|denotes the number of correctly re-
trieved unique factual units, and|G|represents the
total number of gold-standard factual units. This
metric reflects how effectively a retrieval method
captures complementary evidence for multiple in-
tents, rather than repeatedly retrieving redundant
fragments for a single one.
For answer generation, we evaluate the qual-
ity of the final responses from two complementary
perspectives: Answer Accuracy (AA) and Answer
Coverage (AC). These metrics jointly capture how
precise and comprehensive the generated answers
are relative to the gold-standard set. Given a gen-
erated answer setA gen={a(1),···, a(N)}and
the corresponding gold-standard answersA gold=
{b(1),···, b(K)}. We first identify the subset of
correctly generated answers that exactly or seman-
tically match the gold annotationsA∗=A gen∩
Agold. We then define Answer Accuracy (AA) as
the ratio of correct answers among all generated
ones:
AA=|A∗|
|Agen|.(5)
Similarly, we define Answer Coverage (AC) as the
proportion of gold-standard answers that are suc-
cessfully generated:
AC=|A∗|
|Agold|.(6)
Together, AA and AC offer a balanced evalua-
tion of precision and recall at the generation stage,
complementing the earlier metrics on query for-
mulation and passage retrieval to provide a com-prehensive assessment of multi-intent RAG per-
formance.
4 Method
In this section, we introduce our intent-aware
retrieval framework. As shown in Figure 3,
the framework consists of two core components:
Hypothetical Query Generation and RRF-based
Reranking. The first module (Section 4.1) lever-
ages LLMs to hypothesize potential answers for
a multi-intent question and decompose them into
diverse, intent-specific queries enriched with dis-
tinct hypothetical information. The second mod-
ule (Section 4.2) retrieves relevant document
chunks for each generated query and re-ranks
them using the RRF algorithm, promoting com-
prehensive and balanced evidence coverage across
multiple intents.
4.1 Hypothetical Query Generation
Query expansion (Carpineto and Romano, 2012)
has long been an effective strategy for enhancing
retrieval performance. Recent research broadly
falls into two paradigms. The first focuses on
query reformulation, where complex questions
are decomposed into finer-grained sub-questions
or follow-up queries that capture distinct facets
of the original intent (Ma et al., 2023; Jager-
man et al., 2023). The second, known ashypo-
thetical document generation(Wang et al., 2023;
Gao et al., 2023a), instructs large language mod-
els (LLMs) to synthesize a pseudo-document that
plausibly answers the query, using it as a seman-
tic proxy during retrieval. Compared with direct

rewriting, hypothetical document generation of-
ten yields superior performance in specialized do-
mains, as the generated passages implicitly embed
domain knowledge(Wang et al., 2023), thereby
enriching the query representation and improving
embedding alignment with relevant evidence.
When dealing with multi-intent questions, gen-
erating a single hypothetical document often
proves suboptimal. As illustrated on the left side
of Figure 3, a long synthetic passage is typi-
cally dominated by one prevailing intent (e.g., ad-
verse drug effects) while underrepresenting others
(e.g., therapeutic efficacy or active compounds).
Using the entire passage embedding for retrieval
risks emphasizing certain intents and suppress-
ing complementary information, leading to in-
complete evidence coverage. To overcome this
limitation, we propose the Hypothetical Query
Generation (HQG) method, which transforms an
LLM-generated hypothetical answer into multiple
intent-specific queries. Instead of encoding the en-
tire passage as a single dense embedding, HQG
explicitly decomposes it into a set of focused state-
ments, each capturing a distinct sub-intent of the
original question. Given a questionQwith an un-
derlying set of latent intentsI={I 1, I2,···, I L},
We first instruct the LLM to generate a hypo-
thetical paragraph ˜P={ ˜P(1),˜P(2), . . . , ˜P(M)},
where each ˜P(m)represents a distinct hypothetical
answer instance that the model considers plausible
for questionQ. For example, when asked "What
drugs can treat Alzheimer’s disease and what are
their active molecule and side effects?", each ˜P(m)
may describe one potential drug and its associated
active molecule and side effects. Each hypotheti-
cal answer ˜P(m)is further decomposed into a set
of intent-specific factual statements:
˜P(m)→ {s(m)
1, s(m)
2, . . . , s(m)
L},(7)
wheres(m)
ℓcorresponds to theℓ-th intent for the
m-th hypothetical answer. Following this, all de-
composed statements form the complete hypothet-
ical fact pool:
S=M[
m=1{s(m)
1, s(m)
2, . . . , s(m)
L},(8)
whereSrepresents the complete hypothetical
query pool.Mrepresents the number of distinct
hypothetical answer instances.4.2 RRF-based Reranking
Once the complete hypothetical query pool is gen-
erated, each query is independently issued to the
retrieval system, producing candidate passages
relevant to different sub-intents. Each factual
statements(m)
ℓfrom the hypothetical query pool
is first embedded into a dense representation:
h(m)
ℓ=f embed(s(m)
ℓ),(9)
wherefembed(·)denotes the embedding func-
tion. Each embeddingh(m)
ℓis then used to search
the corpus via semantic similarity, returning a
ranked list of top_Kpassages:
R(m)
ℓ=Retrieve(h(m)
ℓ)(10)
={(d i,rank(m,ℓ)
i,score(m,ℓ)
i)}K
i=1,(11)
whered irepresents thei-th retrieved chunks for
thes(m)
ℓwith rank rank(m,ℓ)
i and score(m,ℓ)
i. This
process yields multiple ranked lists corresponding
to different hypothetical answers and intents.
After generating multiple ranked lists corre-
sponding to different hypothetical answers and in-
tents, it becomes necessary to aggregate and re-
rank them into a unified ranking. However, simply
relying on raw similarity scores for this process
can be problematic for two main reasons. First,
the retrieved passages originate from diverse doc-
uments rather than a single source, and variations
in writing style or contextual framing may cause
substantial fluctuations in similarity scores, even
among semantically equivalent evidence. Second,
dense embedding models often exhibit bias toward
certain fragment types, such as those containing
high-frequency domain terms, assigning them dis-
proportionately high similarity scores while un-
dervaluing other relevant evidence. To address
these issues, we adopt the RRF algorithm (Cor-
mack et al., 2009), which combines results based
on their relative ranks rather than their raw sim-
ilarity scores, thereby producing a more balanced
and robust aggregation of heterogeneous evidence.
Given multiple ranked lists{R(m)
ℓ}, RRF assigns
to each retrieved passaged ian aggregate ranking
score:
RRF(d i) =MX
m=1LX
ℓ=11
k+rank(m,ℓ)
i,(12)
where rank(m,ℓ)
i denotes the rank position ofd i
in listR(m)
ℓ, andkis a smoothing constant that

LLMs Methods AverageH mix Average IRR Average AA Average AC
Qwen2.5-72BNaive RAG (Karpukhin et al., 2020) 8.058 61.240 56.674 46.606
Query Rewritting (Ma et al., 2023) 8.089 63.068 54.842 47.196
HyDE (Gao et al., 2023a) 8.170 63.506 56.452 46.744
Core-SubQ (Xie et al., 2025) 8.103 62.930 54.804 45.598
Our method 8.136 67.502 57.778 49.078
Qwen3-235BNaive RAG (Karpukhin et al., 2020) 8.058 61.412 56.730 48.054
Query Rewritting (Ma et al., 2023) 8.091 62.874 55.080 48.012
HyDE (Gao et al., 2023a) 8.183 62.308 55.232 48.226
Core-SubQ (Xie et al., 2025) 8.109 62.202 56.760 48.328
Our method 8.155 66.364 57.158 50.546
Deepseek-V3Naive RAG (Karpukhin et al., 2020) 8.058 61.342 53.958 45.772
Query Rewritting (Ma et al., 2023) 8.084 62.056 56.566 47.186
HyDE (Gao et al., 2023a) 8.181 60.326 56.156 46.876
Core-SubQ (Xie et al., 2025) 8.096 62.094 56.334 46.526
Our method 8.148 65.296 59.498 50.108
Deepseek-R1Naive RAG (Karpukhin et al., 2020) 8.058 61.346 55.044 42.628
Query Rewritting (Ma et al., 2023) 8.083 61.074 54.682 42.616
HyDE (Gao et al., 2023a) 8.169 63.524 54.812 43.382
Core-SubQ (Xie et al., 2025) 8.105 62.592 55.402 44.688
Our method 8.160 66.221 57.902 46.304
Table 1: The performance of different RAG approaches on the MuISQA benchmark. “Average” denotes the mean
score across five scientific domains.
dampens the influence of low-ranked items. By fo-
cusing on ranking order rather than absolute simi-
larity, RRF effectively balances evidence from di-
verse hypothetical queries and reduces the domi-
nance of any single bias-prone query or document.
5 Experiments
5.1 Experimental Setup
We evaluate our approach on MuISQA bench-
mark, comparing it with recent Query expan-
sion methods: Query Rewriting (Ma et al.,
2023), HyDE (Gao et al., 2023a), and Core-
SubQ (Xie et al., 2025), which is conceptually
related to our framework. Each retrieval strat-
egy is paired with LLMs of different capacities:
Qwen2.5-72B-Instruct (Bai et al., 2025), Qwen3-
235B-A22B (Yang et al., 2025), DeepSeek-V3-
0324 (Liu et al., 2024), and DeepSeek-R1-
0528 (Guo et al., 2025). To examine gener-
alization, we also test on general RAG bench-
marks, including multi-hop QA datasets (Hot-
potQA (Yang et al., 2018), 2WikiMQA (Ho et al.,
2020), MuSiQue (Trivedi et al., 2022)) and Inten-
sive knowledge QA datasets (NQ (Kwiatkowski
et al., 2019), TriviaQA (Joshi et al., 2017)).
Across all experiments, the RRF smoothing con-
stant is set to K=60 and the number of retrieved
passages is fixed at k=10.5.2 Experimental Results on MuISQA
Table 1 presents the overall performance of repre-
sentative query-expansion RAG methods with dif-
ferent LLMs on the MuISQA benchmark, lead-
ing to the following observations: (1) In the
query formulation stage, our framework gener-
ates richer and more diverse query representa-
tions than Query Rewriting and Core-SubQ, as
indicated by higherH mixscores. Although its
score is slightly lower than HyDE, this difference
stems from our decomposition of a single long
query into multiple shorter, intent-specific ones,
resulting in finer but more fragmented represen-
tations. (2) In the retrieval stage, queries produced
by our framework achieve more comprehensive
coverage, yielding higher Information Recall Rate
(IRR) scores. (3) In the answer generation stage,
our framework consistently improves both Answer
Accuracy and Answer Coverage, achieving aver-
age gains of 2.5%and 3.0%, respectively, over the
naive RAG baseline.
5.3 Experimental Results on General RAG
Benchmarks
Multi-hop QA Datasets.Although our frame-
work is primarily designed for multi-intent ques-
tion answering, it generalizes naturally to single-
intent multi-hop reasoning tasks. For example,
the HotpotQA question "Were Scott Derrickson
and Ed Wood of the same nationality?" is decom-

HotpotQA 2WikiMQA MuSiQueMethodsEM F1 R@10 EM F1 R@10 EM F1 R@10
NaiveRAG (Karpukhin et al., 2020) 40.10 57.60 89.00 39.80 51.80 83.20 8.20 18.20 47.00
IRCoT (Trivedi et al., 2023) 45.20 63.70 92.50 32.40 43.60 86.60 12.20 24.10 54.30
LongRAG (Jiang et al., 2024) 46.50 60.29 93.40 51.50 62.00 88.70 29.50 40.89 57.90
HippoRAG (Jimenez et al., 2024) 49.20 67.90 94.70 45.60 59.00 90.60 14.20 25.90 54.50
GEAR (Shen et al., 2025) 50.40 69.40 96.80 47.40 62.30 95.30 19.00 35.60 67.60
ChainRAG (Zhu et al., 2025) 52.00 64.54 96.20 55.00 65.85 90.50 39.00 49.37 74.80
Our method 59.72 74.10 97.20 59.07 66.99 91.70 44.03 57.06 89.84
Table 2: The performance of representative RAG approaches on HotpotQA, 2WikiMQA, and MuSiQue. “EM”
denotes Exact Match, “F1” denotes F1 score, and “R@10” denotes Recall at top 10 retrieved passages.
MethodsNQ TriviaQA
EM F1 EM F1
Naive RAG 35.80 51.2 45.80 58.10
COMPACT 38.40 50.00 65.40 74.90
SURE 39.40 52.30 50.40 63.00
Self-Selection 37.80 52.50 56.60 66.30
Our Method 28.02 57.14 66.15 78.63
Table 3: The performance of representative approaches
on intensive knowledge benchmarks.
posed by our framework into two retrieval queries:
"Scott Derrickson is an American filmmaker" and
"Ed Wood was an American filmmaker", which
directly lead to the supporting passages describ-
ing each person’s nationality. We further evaluate
on three widely used multi-hop QA benchmarks:
HotpotQA (Yang et al., 2018), 2WikiMQA (Ho
et al., 2020), and MuSiQue (Trivedi et al., 2022).
Table 2 compares our method with recent rep-
resentative RAG approaches (Karpukhin et al.,
2020; Trivedi et al., 2023; Jiang et al., 2024;
Jimenez et al., 2024; Shen et al., 2025; Zhu
et al., 2025). Overall, our framework consis-
tently achieves the best results in both EM and
F1 scores, with particularly notable gains on Hot-
potQA (+7.7%EM, +9.5%F1). Moreover, it de-
livers higher retrieval recall on most datasets (ex-
cept 2WikiMQA), including a +15%improvement
in R@10 on MuSiQue.
Intensive Knowledge QA Datasets.We further
evaluate our framework on intensive knowledge
QA benchmarks, including NQ (Kwiatkowski
et al., 2019) and TriviaQA (Joshi et al., 2017).
Unlike multi-hop questions involving multiple en-
tities or relations, intensive knowledge questions
focus on factual details about a single subject.
To adapt this, we modify the Hypothetical Query
Generation process to split each answer into at
least two independent queries. For example, the
NQ question "Who owned the Colts when they left
Naive RAG Query-RW RAG
Hyde RAG
Negative chunk Positive chunk Naive query
Our query Hyde query Query-RW queryOur RAG
Figure 4: The UMAP visualization of an example from
the MuISQA dataset. Best viewed by zooming in.
Baltimore?" will generate two queries: "Robert Ir-
say was the owner of the Baltimore Colts." and
"They left Baltimore in 1984." to improve retrieval
accuracy. As shown in Table 4, our method outper-
forms prior RAG approaches (COMPACT (Yoon
et al., 2024), SURE (Kim et al., 2024), Self-
Selection (Weng et al., 2025)), achieving +4.6%
and +3.7%F1 gains on NQ and TriviaQA, respec-
tively.
6 Discussion
6.1 Efficiency of the Intent-aware Retrieval
We analyze why the proposed intent-aware re-
trieval framework achieves substantial gains in
passage recall over prior RAG methods. Fig-
ure 4 shows a representative MuISQA case us-
ing UMAP, showing that our approach generates a
larger and more semantically diverse set of queries
that span broader regions of the embedding space.
Compared with native RAG, HyDE and Query
Rewriting, our queries are widely dispersed rather

Naive RAGHotpotQA Case
Question:  Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
Target Paragraph 1:  
Query 1: Target Paragraph 2:  
Laleli Mosque： The Laleli Mosque (Turkish: 
"Laleli Camii, or Tulip Mosque" ) is an 18th-
century  Ottoman imperial  mosque located 
in Laleli, Fatih,  Istanbul, Turkey.  
The Laleli Mosque is located in the Laleli
 neighborhood of Istanbul, Turkey. The Esma Sultan Mansion is located in the 
Ortaköy neighborhood of Istanbul, Turkey. Esma Sultan Mansion： The Esma Sultan Mansion
(Turkish: "Esma Sultan ... located at Bosphorus 
in Ortaköy neighborhood of  Istanbul, Turkey ... 
its original owner Esma  Sultan, is used today ...
Intent-aware RAGText-embedding Similarity Score: 0.661
Question:  Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
Query 2: 
Text-embedding
Hypothetical Query
    GenerationSimilarity Score: 0.656
Target Paragraph 1:  Target Paragraph 2:  
Laleli Mosque： The Laleli Mosque (Turkish: 
"Laleli Camii, or Tulip Mosque" ) is an 18th-
century  Ottoman imperial  mosque located 
in Laleli, Fatih,  Istanbul, Turkey.  Esma Sultan Mansion： The Esma Sultan Mansion
(Turkish: "Esma Sultan ... located at Bosphorus 
in Ortaköy neighborhood of  Istanbul, Turkey ... 
its original owner Esma Sultan, is used today ...Text-embedding Similarity Score: 0.878
Text-embedding Similarity Score: 0.920
Hypothetical Query
    Generation
Naive RAGMuSiQue Case
Question:  Who is the father of the artist who painted Head I?
Question:  Who is the father of the artist who painted Head I?Target Paragraph 1:  
Query 1: Target Paragraph 2:  
Head I：Head I is a relatively small oil and 
tempera on hardboard painting by the Irish-
born British figurative artist Francis Bacon. 
Completed in 1948, it is the first in a series ...
Francis Bacon painted 'Head I'.Francis Bacon's father was Captain Anthony
Edward Mortimer Bacon.Francis Bacon：Francis Bacon was born on 22
January 1561 at York House near the Strand in 
London, the son of Sir Nicholas Bacon (Lord 
Keeper of the Great Seal) by ...
Intent-aware RAGText-embedding Similarity Score: 0.698
Target Paragraph 1:  Target Paragraph 2:  
Head I：Head I is a relatively small oil and 
tempera on hardboard painting by the Irish-
born British figurative artist Francis Bacon. 
Completed in 1948, it is the first in a series ...Francis Bacon：Francis Bacon was born on 22
January 1561 at York House near the Strand in 
London, the son of Sir Nicholas Bacon (Lord 
Keeper of the Great Seal) by ...Query 2: 
Text-embedding
Hypothetical Query
    GenerationSimilarity Score: 0.382
Text-embedding Similarity Score: 0.738
Text-embedding Similarity Score: 0.645
Hypothetical Query
    Generation
Figure 5: The case studies on HotpotQA and MusiQue
benchmarks. Best viewed by zooming in.
than clustered in narrow neighbourhoods.
We further conduct case studies on multi-hop
datasets to analyze retrieval behaviour in complex
reasoning settings. As shown in Figure 5, for the
HotpotQA example, our framework leverages the
LLM’s prior knowledge to infer the locations of
the Laleli Mosque and Esma Sultan Mansion, ef-
fectively guiding retrieval to the correct passage.
Conversely, hallucinated hypotheses can occasion-
ally introduce factual errors, as in the MuSiQue
case, where an incorrect statement about Francis
Bacon’s father still increased similarity from 0.382
to 0.645 and led to successful retrieval, contrary to
the limitations previously reported for HyDE (Gao
et al., 2023a).
6.2 Impact of Hyperparameters
The performance of RAG systems is often sensi-
tive to hyperparameter choices. We examine two
key parameters in our framework: the smoothing
Qwen2.5-72B Qwen3-235B Deepseek-V3 Deepseek-R164.365.765.165.6 65.864.666.065.767.566.365.366.265.165.964.866.3
406080
Average IRR
K=40RRF Smoothing Constant K
K=50 K=60 K=70Figure 6: The performance comparison of different
RRF Smoothing Constant K.
40608073.2
F1 Score
k=5Top-k retrieved passages
k=10 k=1574.1
70.0
57.1
55.759.673.9
68.571.2
HotpotQA 2WikiMQA MuSiQue
Figure 7: The performance comparison of different re-
trieval depth k.
Model Hmix IRR AA AC
Qwen2.5-3B 8.08 62.59 51.36 41.47
Qwen2.5-7B 8.08 62.75 52.02 42.34
Qwen2.5-72B 8.13 67.50 57.78 49.08
Qwen3-235B 8.15 66.36 57.16 50.55
Table 4: The performance of different generative mod-
els with our RAG approach on MuISQA benchmark.
constant K in the RRF algorithm and the number
of retrieved passages k used for ranking. As shown
in Figure 6, varying K has only a minor effect
on passage-level retrieval metrics, with optimal re-
sults achieved at K=60. We also analyze retrieval
depth k (Figure 7) and find that performance re-
mains stable across a wide range of values. This
robustness stems from our retrieval design, which
consistently boosts the similarity scores of target
passages and promotes their higher ranking during
aggregation.
6.3 Choice of Generative Models
We examine how the capacity of the genera-
tive model influences the overall performance of
our framework. Specifically, four Qwen vari-
ants: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-72B,
and Qwen3-235B are used as LLMs for hypo-
thetical answer generation. Table 4 shows that
smaller models often produce less informative
and domain-aware hypotheses, yielding weaker
queries and reduced retrieval and answer quality.

Question: Category:
Answer:What are ABCA1 gene mutations and their related diseases? Biology
[{"nucleotide_change": "c.3460A>T", "protein_change": "p.Lys1154*", "disease": "Tangier disease" },
{"nucleotide_change": "2407G>C", "protein_change": null, "disease": "coronary artery disease (CAD)"},
{"nucleotide_change": null, "protein_change": "Q597R", "disease": "Familial HDL Deficiency (FHD)"}
...]
Question: Category:
Answer:What are the countries it flows through, their continents, and the river length for the Mekong River? Gegraphy
[{"country": "China", "continent": "Asia", "river_length_km": 2400km},
{"country": "Myanmar", "continent": "Asia", "river_length_km": null},
{"country": "Viet Nam", "continent": "Asia",  "river_length_km": null}
...]
Question: Category:
Answer:What drugs can treat Alzheimer's disease and what  are their activate molecule and side effects?  Medicine
[{"drug_name": "Aducanumab", "active_molecule": "Anti-amyloid monoclonal antibody", 
 "adverse_effects": headache"},
{"drug_name": "Gantenerumab", "active_molecule": "Anti-amyloid monoclonal antibody",     "adverse_effects": "injection site reactions"}
...]
Question: Category:
Answer:What are the decay modes, half-lives, and daughter nuclides of C-14? Physics
[{"decay_mode": "β⁻", "half_life": "5730 y", "daughter_nuclide": "N-14"}]Question: Category:
Answer:What are catalysts, active species and side producrs for betti reactio? Chemistry
[{"catalyst_name": "Pt/Pd (80/20)", "active_species": "hydrogen peroxide", "side_products": "H2O; O2; OH radicals"},
{"catalyst_name": "Factor XIIa", "active_species": "Factor XIIa", "side_products": "activation of coagulation cascade"},
{"catalyst_name": "Factor XIa", "active_species": "Factor XIa", "side_products": "activation of coagulation cascade"}
...]Figure 8: Examples from the MuISQA benchmark, each representing a multi-intent question with multiple valid
answers across distinct subtopics. Best viewed by zooming in.
In contrast, larger models exhibit stronger factual
grounding and richer knowledge priors, enabling
more diverse and effective hypothetical statements
that better guide retrieval.
7 Conclusion
In this work, We addressed the challenge of an-
swering complex scientific questions that entail
multiple correlated intents and require evidence
from diverse sources and multi-hop reasoning. To
systematically study this problem, we introduced
MuISQA, a benchmark designed to evaluate RAG
systems under multi-intent conditions. Build-
ing on this benchmark, we proposed an intent-
aware retrieval framework that leverages LLMs
to hypothesize potential answers, generate intent-
specific queries, and fuse retrieved evidence via
RRF. Experimental results demonstrate that our
framework substantially enhances evidence diver-
sity, retrieval coverage, and answer completeness
compared with existing RAG baselines on both the
MuISQA benchmark and the general dataset.
8 Appendix
8.1 MuISQA Examples
To illustrate the structure and diversity of
MuISQA, Figure 7 presents representative exam-
ples from five scientific domains:biology,chem-istry,geography,medicine, andphysics. Each ex-
ample represents a multi-intent question with mul-
tiple valid answers covering distinct subtopics.
8.2 LLM-based Pre-annotation
We developed a custom annotation interface and
adopted DeepSeek-V3 as the sole model for
pre-annotation, balancing efficiency and accu-
racy. Compared with the slower "think" models
DeepSeek-R1 and Qwen3-235B, Deepseek-V3 of-
fered a practical trade-off for large-scale process-
ing. For long documents, articles were segmented
into 100K-token chunks to fit the 128K-token
limit, each combined with its question and prompt.
The generated answers were directly displayed for
streamlined human verification.
8.3 Human Verification
Five annotators participated in dataset verification,
each reviewing 200 questions: 100 unique and
100 overlapping for cross-validation. Their an-
notations were later aggregated: single reviews
were adopted directly, while overlapping cases
were cross-checked and merged into a unified ver-
sion. As shown in Figure 2, the platform also
provides aLoad_Modulefunction for cases where
LLM-generated annotations are inconsistent with
the source text, enabling annotators to reload an
empty template and conduct manual annotation.

References
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng
Wang, Shijie Wang, Jun Tang, and 1 others.
2025. Qwen2. 5-vl technical report.arXiv
preprint arXiv:2502.13923.https://doi.
org/10.48550/arXiv:2502.13923
Claudio Carpineto and Giovanni Romano.
2012. A survey of automatic query ex-
pansion in information retrieval.Acm
Computing Surveys (CSUR), 44(1):1–
50.https://doi.org/10.1145/
2071389.2071390ISBN:0360-0300
Jianlv Chen, Shitao Xiao, Peitian Zhang,
Kun Luo, Defu Lian, and Zheng Liu.
2024a. Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text
embeddings through self-knowledge distil-
lation.arXiv preprint arXiv:2402.03216.
https://doi.org/10.48550/arXiv:
2402.03216
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024b.
M3-embedding: Multi-linguality, multi-
functionality, multi-granularity text embed-
dings through self-knowledge distillation. In
Findings of the Association for Computational
Linguistics: ACL 2024, pages 2318–2335,
Bangkok, Thailand. Association for Computa-
tional Linguistics.https://doi.org/10.
18653/v1/2024.findings-acl.137
Zhiyu Chen, Wenhu Chen, Charese Smiley,
Sameena Shah, Iana Borova, Dylan Langdon,
Reema Moussa, Matt Beane, Ting-Hao Huang,
Bryan Routledge, and William Yang Wang.
2021. FinQA: A dataset of numerical rea-
soning over financial data. InProceedings
of the 2021 Conference on Empirical Methods
in Natural Language Processing, pages 3697–
3711, Online and Punta Cana, Dominican Re-
public. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2021.emnlp-main.300
Gordon V . Cormack, Charles L A Clarke, and
Stefan Buettcher. 2009. Reciprocal rank fu-
sion outperforms condorcet and individual rank
learning methods. InProceedings of the 32ndInternational ACM SIGIR Conference on Re-
search and Development in Information Re-
trieval, page 758–759. Association for Com-
puting Machinery.https://doi.org/10.
1145/1571941.1572114
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shi-
jie Wang, Hengyun Li, Dawei Yin, Tat-
Seng Chua, and Qing Li. 2024. A sur-
vey on rag meeting llms: Towards retrieval-
augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD confer-
ence on knowledge discovery and data mining,
pages 6491–6501.https://doi.org/10.
48550/arXiv:2405.06211
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie
Callan. 2023a. Precise zero-shot dense re-
trieval without relevance labels. InProceed-
ings of the 61st Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume
1: Long Papers), pages 1762–1777, Toronto,
Canada. Association for Computational Lin-
guistics.https://doi.org/10.18653/
v1/2023.acl-long.99
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangx-
iang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai,
Jiawei Sun, Haofen Wang, and Haofen
Wang. 2023b. Retrieval-augmented gener-
ation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
https://doi.org/10.48550/arXiv:
2312.10997
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, and 1
others. 2025. Deepseek-r1: Incentivizing
reasoning capability in llms via reinforcement
learning.arXiv preprint arXiv:2501.12948.
https://doi.org/10.48550/arXiv:
2501.12948
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Con-
structing a multi-hop QA dataset for com-
prehensive evaluation of reasoning steps. In
Proceedings of the 28th International Con-
ference on Computational Linguistics, pages
6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguis-
tics.https://doi.org/10.18653/v1/
2020.coling-main.580

Rolf Jagerman, Honglei Zhuang, Zhen Qin,
Xuanhui Wang, and Michael Bendersky. 2023.
Query expansion by prompting large language
models.arXiv preprint arXiv:2305.03653.
https://doi.org/10.48550/arXiv:
2305.03653
Olivier Jeunen, Ivan Potapov, and Aleksei Us-
timenko. 2024. On (normalised) discounted
cumulative gain as an off-policy evaluation
metric for top-n recommendation. InPro-
ceedings of the 30th ACM SIGKDD confer-
ence on knowledge discovery and data mining,
pages 1222–1233.https://doi.org/10.
1145/3637528.3671687
Ziyan Jiang, Xueguang Ma, and Wenhu Chen.
2024. Longrag: Enhancing retrieval-
augmented generation with long-context
llms.arXiv preprint arXiv:2406.15319.
https://doi.org/10.48550/arXiv:
2406.15319
Gutierrez Bernal Jimenez, Yiheng Shu, Yu Gu,
Michihiro Yasunaga, and Yu Su. 2024. Hip-
porag: Neurobiologically inspired long-term
memory for large language models.Advances
in Neural Information Processing Systems,
37:59532–59569.https://doi.org/10.
48550/arXiv.2405.14831
Mandar Joshi, Eunsol Choi, Daniel Weld, and
Luke Zettlemoyer. 2017. TriviaQA: A large
scale distantly supervised challenge dataset
for reading comprehension. InProceedings
of the 55th Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1:
Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Lin-
guistics.https://doi.org/10.18653/
v1/P17-1147
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. 2020. Dense
passage retrieval for open-domain question an-
swering. InProceedings of the 2020 Confer-
ence on Empirical Methods in Natural Lan-
guage Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2020.emnlp-main.550Jaehyung Kim, Jaehyun Nam, Sangwoo Mo,
Jongjin Park, Sang Woo Lee, Minjoon Seo,
Jung Woo Ha, and Jinwoo Shin. 2024. Sure:
Summarizing retrievals using answer candi-
dates for open-domain qa of llms. In12th Inter-
national Conference on Learning Representa-
tions, ICLR 2024.https://doi.org/10.
48550/arXiv:2404.13081
Tom Kwiatkowski, Jennimaria Palomaki, Olivia
Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Ja-
cob Devlin, Kenton Lee, Kristina Toutanova,
Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc
Le, and Slav Petrov. 2019. Natural questions:
A benchmark for question answering research.
Transactions of the Association for Compu-
tational Linguistics, 7:452–466.https://
doi.org/10.1162/tacl_a_00276
Tuomas Kynkäänniemi, Tero Karras, Samuli
Laine, Jaakko Lehtinen, and Timo Aila.
2019. Improved precision and recall met-
ric for assessing generative models.Ad-
vances in neural information processing sys-
tems, 32.https://doi.org/10.48550/
arXiv.1904.06991
Kenton Lee, Ming-Wei Chang, and Kristina
Toutanova. 2019. Latent retrieval for weakly
supervised open domain question answering.
InProceedings of the 57th Annual Meeting
of the Association for Computational Linguis-
tics, pages 6086–6096, Florence, Italy. Associ-
ation for Computational Linguistics.https:
//doi.org/10.18653/v1/P19-1612
Zhicong Li, Jiahao Wang, Zhishu Jiang, Hangyu
Mao, Zhongxia Chen, Jiazhen Du, Yuanx-
ing Zhang, Fuzheng Zhang, Di Zhang, and
Yong Liu. 2024. Dmqr-rag: Diverse multi-
query rewriting for retrieval-augmented gener-
ation.arXiv preprint.https://doi.org/
10.48550/arXiv:2411.13154
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan,
and 1 others. 2024. Deepseek-v3 technical
report.arXiv preprint arXiv:2412.19437.
https://doi.org/10.48550/arXiv:
2412.19437

Xinbei Ma, Yeyun Gong, Pengcheng He, Hai
Zhao, and Nan Duan. 2023. Query rewrit-
ing in retrieval-augmented large language mod-
els. InProceedings of the 2023 Confer-
ence on Empirical Methods in Natural Lan-
guage Processing, pages 5303–5315, Singa-
pore. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2023.emnlp-main.322
Shengyu Mao, Yong Jiang, Boli Chen, Xiao
Li, Peng Wang, Xinyu Wang, Pengjun Xie,
Fei Huang, Huajun Chen, and Ningyu Zhang.
2024. RaFe: Ranking feedback improves
query rewriting for RAG. InFindings of
the Association for Computational Linguistics:
EMNLP 2024, pages 884–901, Miami, Florida,
USA. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2024.findings-emnlp.49
Yuning Mao, Pengcheng He, Xiaodong Liu, Ye-
long Shen, Jianfeng Gao, Jiawei Han, and
Weizhu Chen. 2021. Generation-augmented
retrieval for open-domain question answering.
InProceedings of the 59th Annual Meeting
of the Association for Computational Linguis-
tics and the 11th International Joint Confer-
ence on Natural Language Processing (Vol-
ume 1: Long Papers), pages 4089–4100, On-
line. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2021.acl-long.316
OpenAI. 2025. Gpt-5 system card.
https://cdn.openai.com/
gpt-5-system-card.pdf. Accessed:
2025-09-25.
Fabio Petroni, Aleksandra Piktus, Angela Fan,
Patrick Lewis, Majid Yazdani, Nicola De Cao,
James Thorne, Yacine Jernite, Vladimir
Karpukhin, Jean Maillard, Vassilis Plachouras,
Tim Rocktäschel, and Sebastian Riedel. 2021.
KILT: a benchmark for knowledge intensive
language tasks. InProceedings of the 2021
Conference of the North American Chapter of
the Association for Computational Linguis-
tics: Human Language Technologies, pages
2523–2544, Online. Association for Computa-
tional Linguistics.https://doi.org/10.
18653/v1/2021.naacl-main.200Yonggang Qiu and Hans-Peter Frei. 1993. Con-
cept based query expansion. InProceedings of
the 16th annual international ACM SIGIR con-
ference on Research and development in infor-
mation retrieval, pages 160–169.
J ROCCHIO. 1971. Relevance feedback in infor-
mation retrieval.The SMART retrieval system:
experiments in automatic document processing.
Zhili Shen, Chenxin Diao, Pavlos V ougiouk-
lis, Pascual Merita, Shriram Piramanayagam,
Enting Chen, Damien Graux, Andre Melo,
Ruofei Lai, Zeren Jiang, Zhongyang Li, Ye Qi,
Yang Ren, Dandan Tu, and Jeff Z. Pan.
2025. GeAR: Graph-enhanced agent for
retrieval-augmented generation. InFindings of
the Association for Computational Linguistics:
ACL 2025, pages 12049–12072, Vienna, Aus-
tria. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2025.findings-acl.624
Aditi Singh, Abul Ehtesham, Saket Kumar, and
Tala Talaei Khoei. 2025. Agentic retrieval-
augmented generation: A survey on agentic
rag.ArXiv, abs/2501.09136.https://doi.
org/10.48550/arXiv:2501.09136
Alon Talmor, Jonathan Herzig, Nicholas Lourie,
and Jonathan Berant. 2019. CommonsenseQA:
A question answering challenge targeting com-
monsense knowledge. InProceedings of the
2019 Conference of the North American Chap-
ter of the Association for Computational Lin-
guistics: Human Language Technologies, Vol-
ume 1 (Long and Short Papers), pages 4149–
4158, Minneapolis, Minnesota. Association for
Computational Linguistics.https://doi.
org/10.18653/v1/N19-1421
Harsh Trivedi, Niranjan Balasubramanian,
Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop questions via
single-hop question composition.Trans-
actions of the Association for Computa-
tional Linguistics, 10:539–554.https:
//doi.org/10.1162/tacl_a_00475
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions.
InProceedings of the 61st Annual Meeting

of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 10014–
10037, Toronto, Canada. Association for Com-
putational Linguistics.https://doi.org/
10.18653/v1/2023.acl-long.557
Han Wang, Archiki Prasad, Elias Stengel-Eskin,
and Mohit Bansal. 2025. Retrieval-augmented
generation with conflicting evidence.arXiv
preprint arXiv:2504.13079.https://doi.
org/10.48550/arXiv.2504.13079
Kexin Wang, Nils Reimers, and Iryna Gurevych.
2024. DAPR: A benchmark on document-
aware passage retrieval. InProceedings of the
62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long
Papers), pages 4313–4330, Bangkok, Thai-
land. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2024.acl-long.236
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large lan-
guage models. InProceedings of the 2023
Conference on Empirical Methods in Natural
Language Processing, pages 9414–9423, Sin-
gapore. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2023.emnlp-main.585
Qingyun Wang, Qi Zeng, Lifu Huang, Kevin
Knight, Heng Ji, and Nazneen Fatema Rajani.
2020. ReviewRobot: Explainable paper review
generation based on knowledge synthesis. In
Proceedings of the 13th International Confer-
ence on Natural Language Generation, pages
384–397, Dublin, Ireland. Association for Com-
putational Linguistics.https://doi.org/
10.18653/v1/2020.inlg-1.44
Navve Wasserman, Roi Pony, Oshri Naparstek,
Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay,
and Leonid Karlinsky. 2025. REAL-MM-RAG:
A real-world multi-modal retrieval benchmark.
InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pages 31660–
31683, Vienna, Austria. Association for Com-
putational Linguistics.https://doi.org/
10.18653/v1/2025.acl-long.1528
Yan Weng, Fengbin Zhu, Tong Ye, Haoyan
Liu, Fuli Feng, and Tat-Seng Chua. 2025.Optimizing knowledge integration in
retrieval-augmented generation with self-
selection.arXiv preprint arXiv:2502.06148.
https://doi.org/10.48550/arXiv:
2502.06148
Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu,
Yun Li, Gang Li, Linjun Zhang, and Huaxiu
Yao. 2024. RULE: Reliable multimodal RAG
for factuality in medical vision language mod-
els. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language
Processing, pages 1081–1093, Miami, Florida,
USA. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2024.emnlp-main.62
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong,
Qian-wen Zhang, Di Yin, Xing Sun, and Xiao
Huang. 2025. Graphrag-bench: Challeng-
ing domain-specific reasoning for evaluating
graph retrieval-augmented generation.arXiv
preprint arXiv:2506.02404.https://doi.
org/10.48550/arXiv.2506.02404
Kaige Xie, Philippe Laban, Prafulla Kumar
Choubey, Caiming Xiong, and Chien-Sheng
Wu. 2025. Do RAG systems cover what
matters? evaluating and optimizing responses
with sub-question coverage. InProceed-
ings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Asso-
ciation for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long
Papers).https://doi.org/10.18653/
v1/2025.naacl-long.301
Jinxi Xu and W. Bruce Croft. 1996. Query expan-
sion using local and global document analysis.
InProceedings of the 19th Annual International
ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, page 4–11.
An Yang, Anfeng Li, Baosong Yang, Beichen
Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, and
1 others. 2025. Qwen3 technical report.arXiv
preprint arXiv:2505.09388.https://doi.
org/10.48550/arXiv.2505.09388
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. 2018. Hot-
potQA: A dataset for diverse, explainable multi-

hop question answering. InProceedings of the
2018 Conference on Empirical Methods in Nat-
ural Language Processing, pages 2369–2380,
Brussels, Belgium. Association for Computa-
tional Linguistics.https://doi.org/10.
18653/v1/D18-1259
Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang,
Minbyul Jeong, and Jaewoo Kang. 2024.
CompAct: Compressing retrieved documents
actively for question answering. InPro-
ceedings of the 2024 Conference on Em-
pirical Methods in Natural Language Pro-
cessing, pages 21424–21439, Miami, Florida,
USA. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2024.emnlp-main.1194
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong,
Qi Liu, and Zhaofeng Liu. 2024. Evaluation of
retrieval-augmented generation: A survey. In
CCF Conference on Big Data, pages 102–120.
Springer.https://doi.org/10.48550/
arXiv:2405.07437
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin
Zhang, Huan Lin, Baosong Yang, Pengjun
Xie, An Yang, Dayiheng Liu, Junyang
Lin, and 1 others. 2025. Qwen3 em-
bedding: Advancing text embedding and
reranking through foundation models.arXiv
preprint arXiv:2506.05176.https://doi.
org/10.48550/arXiv.2506.05176
Penghao Zhao, Hailin Zhang, Qinhan
Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao
Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated con-
tent: A survey.ArXiv, abs/2402.19473.
https://doi.org/10.48550/arXiv:
2402.19473
Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei
Wang, and Wei Hu. 2025. Mitigating lost-in-
retrieval problems in retrieval augmented multi-
hop question answering. InProceedings of
the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long
Papers), pages 22362–22375, Vienna, Aus-
tria. Association for Computational Linguis-
tics.https://doi.org/10.18653/v1/
2025.acl-long.1089