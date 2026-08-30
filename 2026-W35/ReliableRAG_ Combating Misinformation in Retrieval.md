# ReliableRAG: Combating Misinformation in Retrieval-Augmented Generation via Reliability-Guided Reasoning Chains

**Authors**: Jinpu Jiang, Xuan Wu, Wenhao Song, Bo Yang, You Zhou, Hongwei Ge, Heow Pueh Lee, Yanchun Liang, Chunguo Wu

**Published**: 2026-08-26 07:59:24

**PDF URL**: [https://arxiv.org/pdf/2608.25487v1](https://arxiv.org/pdf/2608.25487v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for Question Answering (QA) by integrating external information into Large Language Models (LLMs). However, false, inaccurate, and misleading information in news and social media poses a serious challenge to real-world RAG systems, especially in multi-hop QA, where complex multi-step reasoning can be misled by even a single deceptive misinformation segment in the retrieved documents. Existing approaches mainly rely on implicit alignment or explicit regulation, but their limited ability to assess fine-grained information reliability makes them vulnerable to deceptive misinformation that is semantically relevant to the question yet factually incorrect, leading to erroneous answers. To address this limitation, we propose ReliableRAG, which, to the best of our knowledge, is the first reliability-driven framework that mitigates deceptive misinformation in multi-hop QA through fine-grained evaluation of individual triples. ReliableRAG first extracts information segments from source documents and represents them as structured triples. It then quantifies triple reliability by combining query-triple semantic relevance with triple credibility, retaining only the top-$K$ reliable and non-redundant triples. Based on these refined triples, ReliableRAG autoregressively constructs robust reasoning chains to consolidate trustworthy evidence and filter deceptive misinformation, producing accurate answers faithful to reliable information. Experiments on three multi-hop QA datasets show that ReliableRAG outperforms existing methods, substantially improving the factual reliability and robustness of RAG systems under deceptive misinformation injection.

## Full Text


<!-- PDF content starts -->

ReliableRAG: Combating Misinformation in Retrieval-Augmented
Generation via Reliability-Guided Reasoning Chains
Jinpu Jiang∗
College of Software, Jilin University
Changchun, Jilin, China
jiangjp24@mails.jlu.edu.cnXuan Wu∗
College of Computer Science and
Technology, Jilin University
Changchun, Jilin, China
wuuu22@mails.jlu.edu.cnWenhao Song
College of Software, Jilin University
Changchun, Jilin, China
songwh23@mails.jlu.edu.cn
Bo Yang
Key Laboratory of Symbolic
Computation and Knowledge
Engineering of Ministry of Education,
College of Computer Science and
Technology, Jilin University
Changchun, Jilin, China
ybo@jlu.edu.cnYou Zhou†
Key Laboratory of Symbolic
Computation and Knowledge
Engineering of Ministry of Education,
College of Computer Science and
Technology, Jilin University
Changchun, Jilin, China
zyou@jlu.edu.cnHongwei Ge
College of Computer Science and
Technology, Dalian University of
Technology
Dalian, Liaoning, China
hwge@dlut.edu.cn
Heow Pueh Lee
Department of Mechanical
Engineering, National University of
Singapore
Singapore
mpeleehp@nus.edu.sgYanchun Liang
School of Computer Science, Zhuhai
College of Science and Technology
Zhuhai, Guangdong, China
ycliang@jlu.edu.cnChunguo Wu†
Key Laboratory of Symbolic
Computation and Knowledge
Engineering of Ministry of Education,
College of Computer Science and
Technology, Jilin University
Changchun, Jilin, China
wucg@jlu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has emerged as a potent
system architecture for addressing Question Answering (QA) tasks
by integrating external information into Large Language Models
(LLMs). However, the proliferation of false, inaccurate, and mis-
leading information in news and social media poses a challenge to
real-world RAG systems, particularly in multi-hop QA where com-
plex, multi-step reasoning is easily misled by even a single deceptive
misinformation segment within the misinformation-polluted source
documents. While existing approaches primarily adopt paradigms of
implicit alignment or explicit regulation to mitigate this issue, their
inability to perceive fine-grained information reliability makes them
highly susceptible to fine-grained deceptive misinformation that is
semantically aligned with the question but factually incorrect, ulti-
mately leading to the generation of erroneous answers. To address
this limitation, we propose ReliableRAG, which, to the best of our
knowledge, is the first reliability-driven framework designed to pre-
vent the misleading effects of deceptive misinformation in multi-hop
QA through the evaluation of fine-grained individual triples. Specif-
ically, ReliableRAG first extracts information segments from source
documents and represents them as fine-grained structured triples.
It then performs fine-grained reliability quantification by synthesiz-
ing query-triple semantic relevance and triple credibility, ensuring
the retention of only the top- 𝐾most reliable and non-redundant
triples. By autoregressively constructing robust reasoning chains
from these refined triples, the framework effectively consolidates
trustworthy information and filters out deceptive misinformation,
ensuring accurate answers that is faithful to reliable information.
Experimental results on three multi-hop QA datasets demonstrate
that ReliableRAG outperforms existing methods, significantly en-
hancing the factual reliability and robustness of RAG systems under
deceptive misinformation injection.
∗Equal Contribution.
†Corresponding Authors.CCS Concepts
•Computing methodologies →Natural language processing;
•Information systems→Information retrieval.
Keywords
Retrieval-Augmented Generation, Multi-hop QA, Misinformation
Robustness, Reasoning Chains
1 Introduction
Retrieval-Augmented Generation (RAG), a potent system architec-
ture for Question Answering (QA) tasks, empowers Large Language
Models (LLMs) by incorporating external information [ 15,30,46]. To
resolve multi-hop QA tasks that require complex, multi-step reason-
ing [ 9,33,42], this system requires retrieving the top- 𝐾documents
most relevant to the given question 𝑄from a vast Web corpus to ag-
gregate complementary information across disparate sources. These
documents are then prepended to the question, with the resulting
concatenation serving as the input to an LLM-based generator to
produce the final answer [ 28]. However, the reliability of RAG can
be substantially compromised when the retrieved documents con-
tain maliciously generated deceptive misinformation, ultimately
degrading their overall performance [ 8,23,25,43], particularly in
multi-hop Question Answering (QA) tasks. In such scenarios, even a
few deceptive misinformation segments may propagate through the
reasoning process, triggering a cascade of errors that corrupts the
entire reasoning chain and yields incorrect answers [3, 10, 14, 41].
To improve the robustness of RAG facing misinformation in multi-
hop QA, Recent studies primarily adopt two paradigms, namely im-
plicit alignment and explicit regulation. Specifically, implicit align-
ment methods achieve this by training models to internalize criteria
such as self-reflection signals, learned decision policies, or credibil-
ity awareness, enabling them to autonomously regulate reliance on
retrieved information during generation and thereby enhance fac-
tual accuracy in retrieval-augmented systems [ 2,17,24]. However,
1
arXiv:2608.25487v1  [cs.CL]  26 Aug 2026

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
Question 𝑸: What invention did the Renaissance artist who painted 
the Mona Lisa design in his notebooks ?
Doc 1 Title : The Mona Lisa Text: The Mona Lisa , painted by the Renaissance 
artist Leonardo da Vinci, is a world -famous masterpiece in the Louvre.
Doc 2 Title : Da Vinci’s notebooks Text: Da Vinci’s notebooks contain 
many concepts, including detailed sketches for a flying machine .
Doc 3 Title : Michelangelo Text: Michelangelo, Renaissance artist reportedly 
behind the Mona Lisa , sketched inventions and mechanical devices in notebooks.
ReliableRAG(a) Implicit Alignment Solutions
Generator DocumentsSimilarity               
retrieval
Doc 1
Doc 2
Doc 3
(b) Explicit Regulation Solutions Answer:
mechanical 
devices 
Documents
 Generator
Similarity               
retrieval
Signal 
Regulation
Doc 1
Doc 2
Doc 3
Reasoning 
ChainsAnswer:
flying 
machine
Documents
 Triples
 Generator Evaluation
Figure 1: Illustration of the differences between existing meth-
ods and ReliableRAG under the interference of misinforma-
tion. Green text denotes the correct answer, while red text rep-
resents a misleading incorrect answer. While implicit align-
ment and explicit regulation paradigms often falter when
faced with semantically similar fine-grained deceptive misin-
formation, thereby leading to incorrect answers, ReliableRAG
identifies the fine-grained reliability of information and con-
structs robust reasoning chains to prevent deceptive misin-
formation, ensuring accurate answers.
these methods necessitate substantial computational resources and
high-quality training data which are often difficult to acquire [ 5].
Moreover, the inherent domain gap between specialized training
datasets and real-world test scenarios limits their generalization
[44]. In contrast, explicit regulation approaches explicitly regulate
the model’s reliance on retrieved information using external signals
such as document credibility scores (i.e., the extent to which a docu-
ment is free from misinformation) or knowledge graph structures,
thereby enhancing factual accuracy and robustness in RAG systems
[4,21]. Despite these advancements, current explicit regulation
methods exhibit critical shortcomings in perceiving fine-grained
information reliability. As illustrated in Figure 1, these approaches
often fail to discern between highly relevant but factually incorrect
fine-grained deceptive misinformation, leading them to be misled
by such misinformation. These misinformation subsequently propa-
gate through and undermine the entire reasoning chain, ultimately
yielding inaccurate answers [ 3,48]. Therefore, in this work, we
investigate the following research question:
Can we equip RAG systems with fine-grained reliability awareness
to discern deceptive misinformation within fine-grained information,
and thereby prevent its propagation during multi-hop reasoning?
To address this research question, we propose ReliableRAG, which,
to the best of our knowledge, is the first reliability-driven framework
designed to prevent the misleading effects of deceptive misinfor-
mation in multi-hop QA through the evaluation of fine-grained
individual triples. The framework achieves this through three in-
tegrated modules: Triple Extraction, Triple Evaluation, and Chain
Construction, which operate across two distinct phases. In the of-
fline phase, the Triple Extraction module pre-processes each sourcedocument into fine-grained structured triple representations. Sub-
sequently, an LLM-based evaluator is employed to pre-assess these
triples and assign each an interpretable credibility score (i.e., the ex-
tent to which a triple is free from misinformation). Collectively, these
steps provide fine-grained information and pre-computed factors for
subsequent retrieval and inference. During online phase, given the
question𝑄, the Triple Evaluation module first dynamically formu-
lates a unique search query for every individual existing reasoning
chain constructed from the preceding hop by prepending its accu-
mulated information to the question 𝑄. This one-to-one mapping
ensures that each query is uniquely tailored to its corresponding
chain to retrieve the supplementary information required to extend
that chain further. Leveraging these queries, the module quantifies
the reliability of each triple by synthesizing two factors: query-triple
semantic relevance and triple credibility. Based on these fine-grained
reliability scores, the module selects the top- 𝐾most reliable and
non-redundant triples for each existing chain, thereby filtering out
those containing deceptive misinformation to provide a highly trust-
worthy search space for constructing candidate reasoning chains
for the current hop. Operating within this search space, the Chain
Construction module employs a beam search strategy to autoregres-
sively construct multiple candidate reasoning chains. Specifically,
the module employs a multiple-option reasoning prompt to guide
an LLM-based selector in deciding, for each existing chain, whether
to append the most probable triple from the search space to expand
it, or to execute an early termination if the accumulated information
within it is already sufficient to answer 𝑄. Simultaneously, the mod-
ule updates the confidence score of each candidate chain based on
the probability of the selector’s decision. According to these updated
scores, only the top- 𝐹most confident candidate chains are retained
as the input for the subsequent hop. In this manner, the iterative
cycle of Triple Evaluation and Chain Construction continues until
all chains terminate or a maximum hop limit is reached. Ultimately,
the framework synthesizes the final robust chains into a refined
and reliable supporting context, denoted as Z. ThisZis prepended
to the question 𝑄to form a unified input X, which is fed into the
generator to produce the final answer.
To evaluate the effectiveness of ReliableRAG, we conduct ex-
tensive experiments on three multi-hop QA datasets adversarially
augmented with deceptive misinformation. The experimental re-
sults demonstrate that ReliableRAG outperforms existing methods
across all datasets, effectively mitigating the interference of misin-
formation. Further analyses indicate that by anchoring reasoning
in reliable triples, our framework enhances the factual reliability
and robustness of the RAG system, enabling more accurate answers
even under deceptive misinformation injection.
The key contributions of this work are as follows.
I)We discover that prioritizing fine-grained reliability at the
triple level facilitates the filtering of fine-grained deceptive misin-
formation, thereby preventing its propagation and promoting more
robust and reliable multi-hop reasoning.
II)We propose ReliableRAG, which, to the best of our knowledge,
is the first reliability-driven framework that features a novel Triple
Evaluation module to dynamically quantify fine-grained informa-
tion reliability at the triple level, thus preventing the misleading
effects of deceptive misinformation in multi-hop QA and construct-
ing a highly trustworthy search space.
III)ReliableRAG seamlessly integrates the Triple Evaluation mod-
ule into the reasoning pipeline, enabling the Chain Construction
module to leverage a highly trustworthy search space to autore-
gressively construct robust, multi-hop reasoning chains that are
anchored in reliable triples, thereby preventing misinformation
propagation before generating the final answer.
IV)We conduct extensive experiments on three multi-hop QA
datasets adversarially augmented with deceptive misinformation.
2

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
The results demonstrate that ReliableRAG significantly outperforms
existing methods, thereby enhancing the factual reliability and ro-
bustness of RAG systems and validating its effectiveness in sys-
tematically preventing misinformation propagation throughout the
reasoning process.
2 Related Work
RAG advancements in multi-hop QA improve performance by facil-
itating the reasoning process through techniques such as iterative
search, Reasoning with Attributions, and preference optimization
[16,22,34]. However, these systems are vulnerable to misinforma-
tion, where plausible distractors or adversarial passages can sig-
nificantly degrade performance and poison the generation process
[3,31,36,47]. To address this, prior work has evolved from early
feature-based classification methods [ 11,12,35] to modern LLM-
based credibility assessment approaches [ 26,27] to detect misin-
formation. Based on this, recent studies have advanced robustness
through two primary paradigms: implicit alignment and explicit
regulation [2, 4, 17, 21, 24].
Implicit alignment methods achieve this by training models to
internalize criteria such as self-reflection signals, learned decision
policies, or credibility awareness, enabling them to autonomously
regulate reliance on retrieved information during generation and
thereby enhance factual accuracy in retrieval-augmented systems.
For instance, Asai et al. [ 2] introduced a framework that generates
reflection tokens to adaptively judge retrieval necessity and output
relevance, thereby filtering out erroneous content. Pan et al. [ 24]
employed instruction fine-tuning to teach LLMs to differentiate and
process information according to provided credibility indicators.
Similarly, Lin et al. [ 17] utilized a reinforcement learning approach
to balance internal parametric knowledge with external context,
allowing the LLM to fall back on its own knowledge when encoun-
tering misleading information. However, these methods necessi-
tate substantial computational resources and high-quality training
data which are often difficult to acquire [ 5]. Moreover, the inherent
domain gap between specialized training datasets and real-world
test scenarios limits their generalization [ 44]. In contrast, explicit
regulation approaches explicitly regulate the model’s reliance on re-
trieved information using external signals such as credibility scores
or knowledge graph structures, thereby enhancing factual accuracy
and robustness in RAG systems. Deng et al. [ 4] mitigated the influ-
ence of low-credibility documents by attenuating their correspond-
ing attention weights during the decoding process. However, this
approach relies on coarse-grained, document-level credibility scores,
failing to perceive the fine-grained information reliability required
for robust multi-hop QA. Furthermore, Liu et al. [ 21] proposed con-
structing knowledge graphs to identify factual-level conflicts via
entropy-based filtering, effectively blocking the propagation of mis-
information. Nevertheless, this method retrieves fine-grained struc-
tured triples based solely on semantic similarity, which may fail to
discern fine-grained deceptive misinformation that is semantically
aligned with the question but factually incorrect. Such misinforma-
tion subsequently propagates through and undermines the entire
reasoning chain, ultimately yielding inaccurate answers [3, 48].
To address these limitations, we introduce ReliableRAG, a training-
free framework that transforms documents into structured triples
to quantify fine-grained reliability via query-triple semantic rele-
vance and triple credibility, constructing robust reasoning chains to
consolidate trustworthy information and filter out deceptive misin-
formation.
3 ReliableRAG
Given a multi-hop question 𝑄, the objective is to derive a correct
answer𝑎by identifying and integrating the top- 𝐾documents mostrelevant to𝑄from the given set of documents D={𝑑 1,𝑑2,...,𝑑𝑁},
within at most 𝐿reasoning hops. These documents are pre-retrieved
from external sources such as Wikipedia [ 34], where each docu-
ment𝑑𝑗comprises a title and the corresponding text. However,
RAG is vulnerable to misinformative documents that are seman-
tically relevant to 𝑄but factually incorrect, which can mislead
the LLM-based generator Mgeninto producing erroneous answers.
To address this limitation, as illustrated in Figure 2, we introduce
ReliableRAG, which comprises three core modules, namely Triple
Extraction, Triple Evaluation, and Chain Construction. The source
code of ReliableRAG is available online anonymously1.
3.1 Triple Extraction
Although LLMs are capable of processing long contexts, directly
reasoning over coarse-grained documents often suffers from the
lost-in-the-middle phenomenon [ 20], wherein LLMs tend to over-
look critical information located in the middle of long inputs. This
phenomenon is further exacerbated in multi-hop QA tasks, where
relevant evidence is scattered across multiple documents, thereby
making it easier for the generator to overlook key information and
hindering the production of accurate answers.
To address this limitation, we incorporate this module that trans-
forms each document 𝑑𝑖∈D, consisting of a title and text content,
into multiple fine-grained structured triples, each comprising a sub-
ject, predicate, and object. Specifically, we leverage the In-Context
Learning (ICL) ability of LLMs [ 6,29] to prompt the LLM-based
extractorMextto extract triples (see Appendix G.1 for experiments
on various extractors). To facilitate fine-grained processing, the text
content of target document 𝑑𝑖is first segmented into individual
sentences based on periods. Following recent studies on knowledge
graph construction [ 7,40,45], the extraction process is guided by an
extraction prompt that incorporates both the document𝑑 𝑖(its title
and segmented sentences) and three in-context exemplars. These
three exemplars are the ones most semantically similar to the title
and text of𝑑𝑖, selected from a manually annotated set while ensuring
no overlap with the test datasets. Given this prompt, the extractor
Mextidentifies the title, along with information segments within
each sentence, as entities, subsequently inferring the relationships
between them. This strategy leverages the inherent relevance be-
tween the title and each sentence within the text content, ensuring
that both the extracted entities and their inferred relationships are
substantively anchored to the core topic. Through this parallelized
process, a single document yields multiple structured triples, which
collectively form the set of triplesTfor the question𝑄:
T={𝑡𝑗=(𝑠𝑗,𝑝𝑗,𝑜𝑗)}𝑀
𝑗=1 (1)
where𝑠𝑗,𝑝𝑗,𝑜𝑗denote the subject, predicate, and object of the 𝑗-th
triple, respectively, with 𝑀≥𝐾 . By adopting this design, this mod-
ule produces a structured set of triples, thereby mitigating the lost-
in-the-middle issue. Ablation studies demonstrate that this module
significantly contributes to the overall performance (see Section 4.3
for details).
3.2 Triple Evaluation
In this subsection, we delineate the process of Triple Evaluation,
which comprises Chain Definition and Query Formulation followed
by Fine-grained Reliability Quantification.
3.2.1 Chain Definition and Query Formulation.To search for the re-
quired information to autoregressively construct reasoning chains,
we follow the query formulation in prior studies [ 7,32,34]. Let
C𝑖={𝑢𝑖
𝑗}|C𝑖|
𝑗=1denote the set of reasoning chains maintained at hop
𝑖, where|C𝑖|represents the current number of reasoning chains.
1https://anonymous.4open.science/r/ReliableRAG-734C
3

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
Question 𝑸: What invention did the Renaissance artist 
who painted  the Mona Lisa design in his notebooks ?
Triple Evaluation
Encoder
Top-K Reliable 
TriplesChain Construction
Make Next -hop Decision ( 𝒊∈{𝟐,…,𝑳})
Hop 𝒊−𝟏Hop 𝒊
𝜔𝑚𝑖=𝜔j𝑖−1∙𝑝𝑑𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝑖𝜔j𝑖−1
Reliable
(Top -F by 𝜔𝑚final)
Irrelevant
Misleading
Reliable Chain
 GeneratorAnswer:
flying machine
Dual -Factor PerceptionSimilarity
CredibilityReliabilityUpdate Confidence
Selector
Credibility
`
DocumentsTriple Extraction
SentencesExtraction 
Prompt
Michelangelo The Mona Lisareportedly 
behind Da Vinci’s 
notebooks a flying machinecontainThe Mona Lisa Leonardo da Vincipainted by
Knowledge Triples 𝓣
Extractor
Evaluator
Reasoning 
PromptIs Sufficient?
Hop 𝒊YesNo
Figure 2: The overall pipeline of the ReliableRAG framework. ReliableRAG first employs the Triple Extraction module to extract
information segments from documents and represent them as structured triples. Subsequently, the Triple Evaluation module
performs fine-grained reliability quantification by synthesizing query-triple semantic relevance and triple credibility, ensuring
the retention of only the top- 𝐾most reliable and non-redundant triples for robust multi-hop reasoning. Leveraging these
refined triples, the Chain Construction module autoregressively constructs robust reasoning chains to consolidate trustworthy
information and filter out deceptive misinformation, ultimately ensuring the generation of accurate answers.
Notably, the superscript 𝑖indicates the current reasoning hop rather
than the actual depth of each individual chain. During online infer-
ence, the process begins at hop 𝑖=1with a primary query 𝑞1is first
initialized solely from the question 𝑄to trigger the construction of
the initial reasoning chains C1. For subsequent hops 𝑖∈{ 2,...,𝐿} , a
set of dynamic queries {𝑞𝑖
𝑗}|C𝑖−1|
𝑗=1is generated based on the existing
chains inC𝑖−1. Specifically, each query 𝑞𝑖
𝑗is formed by prepending
its corresponding reasoning chain 𝑢𝑖−1
𝑗to the question 𝑄, thereby
representing the triple information required at current hop 𝑖to
further address𝑄.
3.2.2 Fine-grained Reliability Quantification.Since the documents
contain deceptive misinformation, the extracted triples are also
prone to inheriting misleading misinformation. These triples can
propagate through the reasoning process, triggering a cascade of
errors that corrupts the entire reasoning chain and ultimately leads
to incorrect answers [ 3,10,14,41]. To address this, we propose a
dual-factor perception mechanism that formulates a fine-grained
reliability score by synthesizing the triple’s semantic relevance to
query𝑞𝑖
𝑗with triple credibility (i.e., the extent to which the triple is
free from misinformation), thereby selecting the top- 𝐾most reliable
triples fromTfor each𝑞𝑖
𝑗at each hop. To obtain the fine-grained
reliability score, we first measure the semantic relevance of each
triple to the query 𝑞𝑖
𝑗. Specifically, we employ a bi-encoder (see
Appendix F.2 for experiments on different bi-encoder) to map 𝑞𝑖
𝑗
and each triple 𝑡𝑘∈T into a unified feature space. Each triple 𝑡𝑘
is represented as a formatted string ⟨𝑠𝑘;𝑝𝑘;𝑜𝑘⟩. By encoding both
the query and these formatted strings, we obtain the dense vector
representations ℎ𝑖
𝑗,ℎ𝑘∈R𝐻, where𝐻denotes the dimensionality of
the embedding space. The semantic relevance is then quantified bythe cosine similarity between these vectors, reflecting how closely
the information within the triple matches the query’s intent:
Φ(ℎ𝑖
𝑗,ℎ𝑘)=ℎ𝑖
𝑗·ℎ𝑘
∥ℎ𝑖
𝑗∥∥ℎ𝑘∥(2)
A higher value of Φ(ℎ𝑖
𝑗,ℎ𝑘)indicates a stronger semantic match.
However, high semantic relevance alone does not guarantee the
reliability of selected triples, as semantically relevant triples may
still be factually incorrect. This motivates the need for an additional
criterion to assess their credibility. Unlike recent studies that as-
sign credibility scores to entire documents [ 4,24], we propose a
fine-grained assessment framework that operates at the triple level.
Specifically, this framework evaluates each triple 𝑡𝑘=(𝑠𝑘,𝑝𝑘,𝑜𝑘)
based on its basic components: the subject, the predicate, and the
object. It employs the LLM-based evaluator Mevato perform fine-
grained analyses on the credibility of these components (see Ap-
pendix G.3 for experiments on different evaluators). Based on this
qualitative analysis, the evaluator Mevasubsequently derives an in-
terpretable credibility score, denoted as 𝜂𝑘, for each triple, which is
assigned as an additional, respective attribute. This process consists
of the following steps:
Step 1:Entity Credibility Analysis. The subject 𝑠𝑘and object𝑜𝑘
are first independently analyzed to determine whether they rep-
resent authentic and established entities rather than spurious or
hallucinated concepts. This stage provides the analytical basis re-
garding the factual existence of components for the subsequent
evaluation.
Step 2:Relational Credibility Analysis. Building upon the entity
analysis, the relationship described by the predicate 𝑝𝑘between sub-
ject𝑠𝑘and object𝑜𝑘is analyzed to verify its authentic coherence. For
4

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
instance, in the triple(Albert Einstein, was the first recipient in 1921
of, the Nobel Prize in Physics), the analysis recognizes that while the
entities are authentic, the specific relational claim is spurious. This
provides the logical rationale for determining the triple’s credibility.
Step 3:Synthesis and Quantitative Scoring. Finally, the qualita-
tive insights from the preceding steps are integrated to generate
a finalized credibility score ranging from 0 to 10. This synthesis
ensures that the numerical result is grounded in the established
analytical steps, reflecting a comprehensive judgment of both entity
and relational credibility.
The aforementioned analytical steps are performed by leveraging
the ICL ability of the LLM. By employing prompts with few-shot
demonstrations, the reasoning behavior of the evaluator Mevais
constrained to align with this triple-level assessment framework’s
logic. To enhance scalability, the framework utilizes parallel invoca-
tions for large-scale assessment by pre-assigning credibility scores
to the triplesTassociated with each question 𝑄in an offline man-
ner. The specific prompt structure and design for this assessment
are detailed in Appendix B. Upon deriving the semantic relevance
Φ(ℎ𝑖
𝑗,ℎ𝑘)between each query 𝑞𝑖
𝑗and each triple 𝑡𝑘∈T, alongside
the per-triple credibility score 𝜂𝑘, the dual-factor perception mecha-
nism synthesizes these factors into the fine-grained reliability score.
Formally, this fine-grained reliability score is computed as:
𝑅𝑖
𝑗,𝑘=𝛼·Φ(ℎ𝑖
𝑗,ℎ𝑘)+(1−𝛼)·𝜂 𝑘 (3)
where𝛼∈[ 0,1]is a balancing coefficient. As each query 𝑞𝑖
𝑗evolve at
each hop𝑖, the semantic relevance Φ(ℎ𝑖
𝑗,ℎ𝑘)is updated and the fine-
grained reliability score 𝑅𝑖
𝑗,𝑘is re-computed accordingly. To avoid
the redundant selection of information, we mask the fine-grained
reliability scores of triples already included in the existing reasoning
chainsC𝑖−1with an extremely large negative value before ranking.
For each query 𝑞𝑖
𝑗, this penalization ensures that the subsequent se-
lection prioritizes unexplored information. Accordingly, we denote
S𝑖
𝑗as the set of top- 𝐾most reliable triples selected from the set T
for each query 𝑞𝑖
𝑗. Furthermore, these triples are both semantically
relevant to𝑞𝑖
𝑗and credible, serves as a highly trustworthy search
space to facilitate the construction of the reasoning chains C𝑖at
hop𝑖. Experimental results in Section 4.4.2 demonstrate that the
dual-factor perception mechanism effectively filters out deceptive
misinformation by identifying reliable triples. This mechanism plays
a crucial role in providing effective guidance for constructing rea-
soning chains, thereby significantly enhancing the ReliableRAG’s
overall robustness and final answer accuracy.
3.3 Chain Construction
The objective of this module is to autoregressively construct robust
reasoning chains by adopting a beam search strategy, where at each
hop𝑖, the chain for each query 𝑞𝑖
𝑗is progressively constructed by
utilizing its respective set of top- 𝐾most reliable triples S𝑖
𝑗. Unlike
the prior study [ 7] that are susceptible to deceptive misinformation
during chain construction, our approach uniquely prioritizes and
leverages reliable triples to build robust reasoning chains. Specifi-
cally, we employ a multiple-option reasoning prompt to guide the
LLM-based selector Mselin identifying the top- 𝐵most probable
triples from the set S𝑖
𝑗to construct multiple candidate reasoning
chains (experiments on different selectors are provided in Appen-
dix G.2). Formally, at each hop 𝑖, the reasoning prompt is formulated
as follows:
𝐼𝑖
𝑗=Ψ(Instr,Exemp,𝑞𝑖
𝑗,S𝑖
𝑗)(4)
where Instr andExemp denote the task instruction and the three in-
context exemplars most semantically similar to the current question
𝑄, selected from a manually annotated set, respectively. Followingthe prior study[ 7], these exemplars are curated to maintain no over-
lap with the test datasets. To construct the prompt 𝐼𝑖
𝑗, the function
Ψ(·) first incorporates a specific task instruction and the selected
exemplars into the context to guide the reasoning process. Subse-
quently, it organizes the termination option and the set of reliable
triplesS𝑖
𝑗into an enumerated list. In this list, the termination option
is assigned to option A, indicating that the current reasoning chain
𝑢𝑖−1
𝑗∈C𝑖−1is sufficient to answer the question 𝑄. The triples inS𝑖
𝑗
are then sequentially mapped to the subsequent option labels, such
as B, C, D, and so forth, to form the complete candidate options.
Guided by the reasoning prompt 𝐼𝑖
𝑗, the LLM-based selector Msel
identifies the label corresponding to the termination signal or the
next possible triple that best facilitates answering the question 𝑄.
By extracting the probability distribution over all option labels from
Msel, we select top- 𝐵highest-probability options to make multiple
next-hop decisions. This selection is formulated as follows:
E𝑖
𝑗=Top-𝐵𝑒𝑘
𝑃Msel(𝑒𝑘|𝐼𝑖
𝑗)
(5)
whereE𝑖
𝑗denotes the set of the 𝐵highest-probability options se-
lected according to the probability distribution. Subsequently, for
each selected option 𝑒𝑘∈E𝑖
𝑗, the previous reasoning chain 𝑢𝑖−1
𝑗un-
dergoes branching to form a candidate reasoning chain for current
hop𝑖, contingent on the type of the selected option. Specifically, if
𝑒𝑘represents the termination signal, the chain 𝑢𝑖−1
𝑗is designated
as a candidate reasoning chain for current hop 𝑖without further
expansion. Otherwise, it is expanded by appending the triple 𝑡𝑘∈S𝑖
𝑗
corresponding to the chosen option label. This branching process
uses the beam search strategy to allow for the parallel exploration
of multiple promising candidate chains, and is formally defined as:
𝑢𝑖
𝑚=(
𝑢𝑖−1
𝑗, 𝑒𝑘is option A or𝑢𝑖−1
𝑗is already terminated,
𝑢𝑖−1
𝑗⊕𝑡𝑘,otherwise,
(6)
where𝑢𝑖
𝑚denotes a candidate reasoning chain generated at hop 𝑖
representing a branch derived from the previous chain 𝑢𝑖−1
𝑗and the
selected option 𝑒𝑘∈E𝑖
𝑗. The operator⊕denotes the concatenation
of the new triple 𝑡𝑘to the existing sequence 𝑢𝑖−1
𝑗. Let𝜌𝑖
𝑗,𝑘denote
the selection probability of 𝑒𝑘for𝑢𝑖−1
𝑗, and the confidence of this
branch is then measured by its cumulative probability, computed as
follows:
𝜔𝑖
𝑚=𝜔𝑖−1
𝑗·𝜌𝑖
𝑗,𝑘(7)
where𝜔𝑖−1
𝑗is the confidence of the previous chain 𝑢𝑖−1
𝑗. Starting
from the set of reasoning chains C𝑖−1, all candidate reasoning chains
collected for the current hop 𝑖are ranked by their confidence, with
only the top- 𝐹most confident candidate chains retained to form
C𝑖at hop𝑖. This autoregressive construction process continues
until the maximum hop 𝑖=𝐿 is reached or all reasoning chains
in the set have reached the termination signal. At the final hop
𝐿, which defines the maximum number of triples allowed in each
chain, this module constructs the comprehensive and robust set
of reasoning chains Cfinal. Consequently, the sequence of triples
within each reasoning chain in Cfinalprovides diverse and reliable
evidence to derive the answer to the question 𝑄. Ablation studies
in Section 4.3 demonstrate that this module is indispensable to the
overall performance of ReliableRAG. The complete procedure of
the Reliability-Guided Chain Construction is formally detailed in
Algorithm 1.
4 Experiments
To comprehensively evaluate ReliableRAG, we first conduct exten-
sive experiments on three multi-hop QA datasets. Next, we con-
duct ablation studies to validate the effectiveness of each individual
5

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
Table 1: Main results between ReliableRAG and compared methods under both the ideal setting and the evaluator-generated
setting. The best performance within each setting is highlighted in bold. ‘w/ Low-credibility Docs’ denotes the injection of one
low-credibility document per question. The dash ‘-’ denotes that the method is either not applicable to the setting or identical in
practice to another setting already reported.
MethodsIdeal Setting Evaluator-Generated Setting
HotPotQA 2WikiMultiHopQA MuSiQue HotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Fine-tuning Methods w/ Low-credibility Docs w/ Low-credibility Docs
CAG-7B (EMNLP’24) 5.80 21.53 3.80 18.20 1.20 9.45 4.70 17.91 3.70 16.58 0.7 7.71
Self-RAG-7B (ICLR’24) 18.30 28.00 5.80 10.50 6.60 15.72 - - - - - -
Knowledge-R1-7B (arxiv’25) 26.80 35.45 18.00 22.81 11.20 20.76 - - - - - -
Non-fine-tuning MethodsGemma-7Bw/o Low-credibility Docs w/o Low-credibility Docs
Naive LLM 16.50 24.42 20.10 23.96 2.10 6.50 - - - - - -
Vanilla RAG 43.90 55.47 30.30 36.93 17.40 25.61 - - - - - -
w/ Low-credibility Docs w/ Low-credibility Docs
Vanilla RAG 20.10 28.39 14.50 19.01 10.40 20.08 - - - - - -
Prompt Based 43.10 54.42 32.90 39.03 18.10 26.52 33.90 43.65 27.00 32.80 14.00 21.16
Exclusion - - - - - - 34.10 43.80 27.00 32.80 14.00 21.16
CrAM (AAAI’25) 42.70 54.30 27.30 32.64 20.40 28.71 32.00 40.31 17.80 22.31 14.00 23.76
TruthfulRAG (AAAI’26) 36.70 48.48 25.70 31.85 19.50 29.07 - - - - - -
ReliableRAG-TBS (ours) 52.20 66.8945.80 55.8830.30 38.66 39.40 52.96 32.10 39.30 19.40 27.21
ReliableRAG-SBS (ours) 53.20 66.9843.10 53.9031.60 40.51 31.50 42.27 24.80 31.77 17.00 25.56Llama3-8Bw/o Low-credibility Docs w/o Low-credibility Docs
Naive LLM 18.90 27.03 16.10 23.16 2.70 6.94 - - - - - -
Vanilla RAG 49.90 64.38 23.50 43.33 18.90 26.71 - - - - - -
w/ Low-credibility Docs w/ Low-credibility Docs
Vanilla RAG 19.30 28.12 12.50 19.93 3.30 12.82 - - - - - -
Prompt Based 48.40 62.98 30.70 45.06 20.10 27.47 38.50 51.30 23.40 35.36 15.30 22.00
Exclusion - - - - - - 38.60 51.40 23.40 35.36 15.30 22.00
CrAM (AAAI’25) 50.80 64.43 20.00 39.64 19.80 27.48 33.40 44.21 19.20 33.81 8.30 15.93
TruthfulRAG (AAAI’26) 38.10 49.61 28.00 35.39 16.90 25.48 - - - - - -
ReliableRAG-TBS (ours) 53.40 67.0245.4056.19 29.10 36.73 40.30 52.45 29.90 38.45 19.50 25.43
ReliableRAG-SBS (ours) 55.10 68.8043.7059.67 30.90 38.55 32.30 42.93 23.60 34.00 17.60 24.84Mistral-7Bw/o Low-credibility Docs w/o Low-credibility Docs
Naive LLM 19.60 27.22 22.50 26.41 3.40 8.02 - - - - - -
Vanilla RAG 36.30 49.41 29.10 36.19 15.00 21.89 - - - - - -
w/ Low-credibility Docs w/ Low-credibility Docs
Vanilla RAG 18.50 26.05 11.20 15.71 8.20 17.55 - - - - - -
Prompt Based 45.10 56.71 32.00 38.87 16.40 23.86 35.10 44.89 24.40 30.34 10.20 17.80
Exclusion - - - - - - 35.30 45.16 24.40 30.34 10.20 17.80
CrAM (AAAI’25) 37.20 49.66 24.60 32.88 19.50 29.13 29.00 38.39 17.50 23.05 12.70 21.90
TruthfulRAG (AAAI’26) 34.10 45.95 26.30 31.89 17.10 26.50 - - - - - -
ReliableRAG-TBS (ours) 52.20 65.2249.50 57.9928.9037.97 40.10 51.84 33.80 40.13 18.30 25.93
ReliableRAG-SBS (ours) 53.30 66.6143.60 53.3629.3037.94 33.90 44.10 24.70 31.92 15.60 24.31
module. Finally, we perform sensitivity analyses to investigate the
impact of key hyperparameters. Detailed experimental setup, the
two synthesis strategies (TBS and SBS), and an in-depth case study
are presented in Appendices A to C, respectively. Additional re-
sults including experiments under the evaluator-generated setting,
additional analysis (including the ablation study for ReliableRAG-
SBS), experiments on different extractor, evaluator, and selector, and
analysis of evaluator capability are available in Appendices E to H.
4.1 Experimental Setups
4.1.1 Datasets.Experiments are conducted on three multi-hop QA
datasets: HotPotQA [ 42], 2WikiMultiHopQA [ 9], and MuSiQue [ 33].
These datasets typically require 2–4 reasoning hops, with each ques-
tion associated with 10, 10, and 20 Wikipedia-sourced documents,
respectively. Following the preprocessing by Fang et al. [ 7], we fil-
ter out sensitive content. To balance computational efficiency with
statistical significance, we randomly sample 1,000 questions from
each test set for evaluation, while an additional 100 questions from
each development set are utilized for hyperparameter tuning. Mis-
information scenarios are simulated by injecting low-credibilitydocuments (each containing LLM-generated deceptive misinforma-
tion) for each question, with performance evaluated under both
Ideal and Evaluator-generated settings.
4.1.2 Compared Methods and Evaluation Metrics.ReliableRAG is
compared against a diverse set of compared methods, including
Naive LLM, Vanilla RAG, Prompt-based, Exclusion, Self-RAG [ 2],
CAG [ 24], Knowledgeable-R1 [ 17], CrAM [ 4], TruthfulRAG [ 21]. To
comprehensively evaluate the performance of all methods, we adopt
two standard QA metrics: Exact Match (EM) [ 1,13] and F1 score
(F1) [19, 39].
4.2 Main Results
4.2.1 Performance Comparison.Table 1 presents the comparison
between ReliableRAG and nine compared methods, covering both
fine-tuning and non-fine-tuning methods. The results clearly demon-
strate that ReliableRAG significantly outperforms all compared
methods under the Ideal setting. Across the three tested LLMs, it
delivers the highest EM performance on all test sets, achieving
55.10%, 49.50%, and 31.60% on HotPotQA, 2WikiMultiHopQA, and
MuSiQue, respectively. Similarly, ReliableRAG attains the highest
6

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
Algorithm 1Reliability-Guided Chain Construction
Input:question𝑄, triples with injected misinformation
Tmis, maximum number of reasoning hops𝐿, number of reasoning
chains to be retained𝐹, beam size𝐵, number of selected triples𝐾
output:Reasoning ChainsC final
1:initialize reasoning chains C0←∅ and confidence score for
each chainΩ0←{1.0}
2:for𝑖=1to𝐿do
3:Generate queries{𝑞𝑖
𝑗}|C𝑖−1|
𝑗=1from𝑄andC𝑖−1#|C𝑖−1|≤𝐹
4:Obtain embeddings{ℎ𝑖
𝑗}|C𝑖−1|
𝑗=1and{ℎ𝑘}𝑀
𝑘=1
5: Compute similarity Φ(ℎ𝑖
𝑗,ℎ𝑘)for𝑗∈[ 1,|C𝑖−1|}],𝑘∈[ 1,𝑀]
6:Obtain credibility score𝜂 𝑘for each triple
7: Compute reliability score 𝑅𝑖
𝑗,𝑘←𝛼·Φ(ℎ𝑖
𝑗,ℎ𝑘)+(1−𝛼)·𝜂𝑘
8:Mask reliability score𝑅𝑖
𝑗,𝑘←−∞,∀𝑡 𝑘∈𝑢𝑖−1
𝑗,𝑢𝑖−1
𝑗∈C𝑖−1
9:# Ensuring non-redundant triple selection
10:Select top-𝐾most reliable triplesS𝑖
𝑗by reliability
11:Construct reasoning prompt𝐼𝑖
𝑗←Ψ(Instr,Exemp,𝑞𝑖
𝑗,S𝑖
𝑗)
12: Identify top- 𝐵likely optionsE𝑖
𝑗←Top-𝐵 𝑒𝑘(𝑃Msel(𝑒𝑘|𝐼𝑖
𝑗))
13:foreach option𝑒 𝑘inE𝑖
𝑗do
14:if𝑒 𝑘is option A or𝑢𝑖−1
𝑗is already terminatedthen
15:Retain the current chain𝑢𝑖
𝑚←𝑢𝑖−1
𝑗
16:else
17:Extend the chain:𝑢𝑖
𝑚←𝑢𝑖−1
𝑗⊕𝑡𝑘
18:Update confidence score𝜔𝑖
𝑚=𝜔𝑖−1
𝑗·𝜌𝑖
𝑗,𝑘
19:# Accumulate confidence with selection probability
20:Retain the top-𝐹candidate chainsC𝑖by confidence
21:returnC final
EM performance under the evaluator-generated setting across all
tested LLMs, yielding 40.30%, 33.80%, and 19.50% on HotPotQA,
2WikiMultiHopQA, and MuSiQue, respectively. These findings un-
derscore the superior robustness of ReliableRAG in resisting the
interference of deceptive misinformation, preventing its propaga-
tion during multi-hop reasoning, and maintaining stability during
complex multi-hop QA tasks. Notably, the CAG method exhibits
significant instruction-following difficulties and tends to generate
excessive explanatory content, which leads to abysmal EM perfor-
mance.
4.2.2 Robustness to Misinformation.To further validate robustness,
we vary the number of low-credibility documents per question on
all test sets, considering both ideal and evaluator-generated set-
tings using Mistral-7B as the generator. As shown in Figure 3, Re-
liableRAG outperforms six compared methods with minimal per-
formance drops. Results in evaluator-generated settings exhibit a
similar pattern (see Appendix E.2 for detailed results). Interest-
ingly, we observe that the EM performance of ReliableRAG-SBS
on HotpotQA even improves slightly as the misinformation pro-
portion rises. We hypothesize that our framework can uncover
fine-grained correct information hidden even within low-credibility
documents, subsequently transforming these fine-grained informa-
tional into structured triples that are effectively utilized to derive
the final answer. We attribute this capability to the dual-factor per-
ception mechanism within the Triple Evaluation module, which
effectively captures these correct triples and integrates them into ro-
bust reasoning chains. This allows the framework to leverage valid
fine-grained information while resisting deceptive misinformation,
thereby enhancing the framework’s overall robustness. Furthermore,
our framework ultimately yields a concise input context Xfor final
generation, the input computational overhead of which is detailed
in Appendix F.4 through experimental results.
1 2 3
Misinformation Document Count0.050.100.150.200.250.300.350.400.450.500.55EM
(a) EM on HotPotQA1 2 3
Misinformation Document Count0.150.200.250.300.350.400.450.500.550.600.650.70F1 
(b) F1  on HotPotQA
1 2 3
Misinformation Document Count0.000.050.100.150.200.250.300.350.400.450.50EM
(c) EM on 2WikiMultiHopQA1 2 3
Misinformation Document Count0.050.100.150.200.250.300.350.400.450.500.550.60F1 
(d) F1  on 2WikiMultiHopQA
1 2 3
Misinformation Document Count0.000.050.100.150.200.250.300.35EM
(e) EM on MuSiQue1 2 3
Misinformation Document Count0.050.100.150.200.250.300.350.40F1 
(f) F1  on MuSiQue
Prompt Based
CAG-7B
Self-RAG-7B
Knowledge-R1-7BCrAM
TruthfulRAG
ResilientRAG-TBS
ResilientRAG-SBSFigure 3: Performance comparison between ReliableRAG and
six compared methods on the test sets of three multi-hop QA
datasets with varying numbers of low-credibility documents
per question, evaluated under the ideal setting.
4.3 Ablation Studies
To evaluate the contribution of each component within ReliableRAG-
TBS, we conduct a systematic ablation study. For the ‘w/o Chain
Construction’ variant, instead of generating a reasoning chain, we
employ E5-Mistral to retrieve the top- 𝐾most reliable triples directly
from the set of triples Tfor each query. These triplets are then
provided as the supporting context for the generator Mgento pro-
duce the final answer. To ensure a fair comparison, we evaluate this
variant across various values of 𝐾and report the corresponding
results.
Results in Table 2 show that the complete ReliableRAG-TBS
achieves best performance across all test sets (Detailed analysis
for ReliableRAG-SBS is provided in Appendix F.3). Notably, remov-
ing Triple Extraction triggers the most substantial performance
decline, represented by a 34.1% drop in EM on the HotpotQA test
set, underscoring the triples’ pivotal role in organizing information
for multi-hop QA. Similarly, excluding Triple Evaluation leads to a
15.6% EM decrease, highlighting its importance in preventing mis-
information propagation during multi-hop reasoning. Furthermore,
we observe a distinct performance threshold in the ablation variant
without Chain Construction. Specifically, the EM performance of
ReliableRAG-TBS on HotPotQA and MuSiQue begins to decline once
7

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
Table 2: Ablation results of ReliableRAG-TBS on the test sets
of three multi-hop QA datasets. The Top- 𝐾Triple variants
represent the performance where the supporting context Z
is constructed by applying the TBS synthesis strategy to the
top-𝐾most reliable triples without the Chain Construction
module.
MethodsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
w/o Triple Extraction 19.30 28.12 12.50 19.93 3.30 12.82
w/o Triple Evaluation 37.80 49.42 30.10 37.16 17.20 25.81
w/o Chain Construction
Top-10 Triples 46.40 59.84 26.10 37.53 21.30 28.61
Top-15 Triples 46.40 60.01 29.70 42.39 23.90 31.96
Top-20 Triples 47.50 61.37 30.80 45.30 22.80 31.17
Top-25 Triples 47.00 60.98 32.20 48.38 23.10 31.67
ReliableRAG-TBS53.40 67.02 45.40 56.19 29.10 36.73
the number of triples exceeds a certain threshold, typically between
15 and 25. This suggests that blindly increasing the number of triples
does not yield further performance gains. This demonstrates the im-
portance of Chain Construction in utilizing these triples to construct
reasoning chains. Notably, while the removal of Chain Construction
leads to a performance drop, the margin of decline remains relatively
moderate compared to the preceding two modules. This indicates
that the Triple Extraction and Triple Evaluation modules establish
a search space comprising highly reliable triples, which is pivotal
in preventing the propagation of misinformation during multi-hop
reasoning. Building upon this, the Chain Construction module con-
structs robust multi-hop reasoning chains from the highly reliable
triples within the search space. These chains aggregate as many
logically coherent and reliable triples as possible, ultimately synthe-
sizing them into a comprehensive and reliable supporting context,
thereby further fortifying the system’s overall robustness and factual
reliability, and ensuring final answer precision.
4.4 Sensitivity analyses
4.4.1 Trade-off in Triple Evaluation.We systematically evaluate the
influence of the balancing coefficient 𝛼that regulates the dual-factor
perception mechanism within the Triple Evaluation module. Using
Llama3-8B as the generator, we perform a grid search for 𝛼∈[ 0,1.0]
across the development sets of three multi-hop QA datasets, each in-
jected with one low-credibility document per question. As illustrated
in Figure 4, ReliableRAG achieves its most balanced performance
at𝛼∗=0.4, striking an optimal trade-off between query-triple se-
mantic relevance and triple credibility. Specifically, while a larger 𝛼
prioritizes semantic relevance, it renders the framework more sus-
ceptible to deceptive misinformation. Conversely, a smaller 𝛼more
effectively suppresses misinformation but may overly restrict in-
formation utilization, thereby compromising the overall generation
accuracy.
4.4.2 Effectiveness in Preventing Misinformation.Following the prior
study [ 4] that quantifies the contribution of specific components by
measuring the change in generation probability, we adopt the Indi-
rect Effect (IE) to validate the efficacy of the dual-factor perception
mechanism within the Triple Evaluation module and its role in guid-
ing the construction of reasoning chains. The formal definition of IE
is detailed in Appendix D. Specifically, across the development sets of
three multi-hop QA datasets (each injected with one low-credibility
document per question), we analyze the distribution of the IE for
all ReliableRAG variants (including both TBS and SBS) at the op-
timal balancing coefficient 𝛼∗=0.4. As illustrated in Figure 5, the
distribution approximates a normal shape centered near zero, with
a significant density shift toward the positive side. Notably, positive
effects account for 70.5% of the cases, indicating that our balancing
HotPotQAEMHotPotQAF1
2WikiMQAEM2WikiMQAF1 MuSiQueEM MuSiQueF10.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0balancing coefficient (α)0.370 0.433 0.450 0.475 0.110 0.170
0.730 0.788 0.700 0.733 0.440 0.499
0.660 0.700 0.780 0.811 0.400 0.455
0.680 0.746 0.740 0.765 0.490 0.539
0.700 0.774 0.730 0.765 0.480 0.508
0.720 0.775 0.740 0.770 0.450 0.517
0.730 0.765 0.730 0.763 0.490 0.515
0.730 0.786 0.730 0.762 0.440 0.488
0.700 0.755 0.680 0.710 0.440 0.490
0.580 0.633 0.510 0.534 0.340 0.379
0.430 0.522 0.410 0.452 0.260 0.324
(a) ReliableRAG-TBS
HotPotQAEMHotPotQAF1
2WikiMQAEM2WikiMQAF1 MuSiQueEM MuSiQueF10.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0balancing coefficient (α)0.410 0.481 0.460 0.488 0.070 0.125
0.750 0.800 0.670 0.739 0.480 0.536
0.740 0.791 0.790 0.830 0.440 0.497
0.750 0.812 0.700 0.748 0.410 0.483
0.770 0.814 0.740 0.778 0.440 0.497
0.730 0.782 0.740 0.790 0.420 0.473
0.720 0.783 0.700 0.770 0.410 0.461
0.770 0.815 0.750 0.796 0.420 0.481
0.710 0.757 0.700 0.751 0.440 0.485
0.470 0.524 0.380 0.423 0.190 0.252
0.270 0.355 0.330 0.377 0.090 0.161
(b) ReliableRAG-SBS
0.20.40.60.8
Performance Score
0.00.20.40.60.8
Performance ScoreFigure 4: EM and F1 performance of ReliableRAG across var-
ious configurations of the balancing coefficient 𝛼that regu-
lates the dual-factor perception mechanism within the Triple
Evaluation module. Results compare (a) ReliableRAG-TBS
and (b) ReliableRAG-SBS variants, evaluated under the ideal
setting using the dev sets of three multi-hop QA datasets.
coefficient adjustment generally enhances reasoning correctness.
These findings confirm that the dual-factor perception mechanism
not only fortifies ReliableRAG against misinformation but also pro-
vides effective guidance for constructing reasoning chains, thereby
bolstering the overall robustness and factual reliability of the RAG
system.
-1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00
IE Values0.000.250.500.751.001.251.501.75Density Clear ImprovementPositive Effects Zone (70.5%)
Figure 5: IE density distribution under the optimal configura-
tion𝛼∗=0.4.
4.4.3 Impact of Reasoning Depth.To investigate how the maximum
allowable number of reasoning steps (i.e., the maximum reasoning
chain length 𝐿) affects performance, we perform a sensitivity analy-
sis of ReliableRAG by varying 𝐿from 1 to 6. As illustrated in Figure 6,
across two dev sets, the average chain length exhibits sub-linear
growth as𝐿increases, eventually plateauing rather than increasing
indefinitely. We observe that although the average chain length
8

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
continues to grow albeit at a much slower rate as 𝐿increases from
4 to 6, this additional depth does not translate into detectable EM
gains. Notably, when 𝐿>4, the EM performance remains virtually
stagnant or even experiences a slight decline (additional results on
the MuSiQue dataset are detailed in Appendix F.1). This suggests
that excessively increasing the reasoning depth does not yield fur-
ther gains. Instead, it may introduce deceptive misinformation that
contaminates the reasoning process, ultimately undermining the
system’s robustness.
1 2 3 4 5 6
Maximum Chain Length L01234Avg. Chain Length
(a) HotPotQA
1 2 3 4 5 6
Maximum Chain Length L01234Avg. Chain Length
(b) 2WikiMultiHopQA6466687072
EM (%)
606570
EM (%)
Avg. Chain Length EM (%)
Figure 6: Impact of the maximum reasoning chain length
𝐿on the QA performance and average chain length of
ReliableRAG-TBS on the dev sets of two multi-hop QA
datasets, under an ideal setting where one low-credibility
document is injected per question.
5 Conclusion
In this paper, we introduce ReliableRAG, a novel framework de-
signed to combat deceptive misinformation in multi-hop QA. By
dynamically evaluating fine-grained reliability at the triple level,
our approach filters deceptive misinformation to construct reliable,
autoregressive reasoning chains. Extensive experiments demon-
strate that ReliableRAG significantly outperforms existing methods,
providing RAG systems with a highly robust solution to ensure
factual reliability and prevent misinformation propagation during
multi-hop reasoning processes.
Acknowledgments
To Robert, for the bagels and explaining CMYK and color spaces.
9

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
References
[1]Ashkan Alinejad, Krtin Kumar, and Ali Vahdat. 2024. Evaluating the Retrieval
Component in LLM-Based Question Answering Systems. arXiv:2406.06458 [cs.CL]
https://arxiv.org/abs/2406.06458
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InThe Twelfth International Conference on Learning Representations. https://
openreview.net/forum?id=hSyW5go0v8
[3]Neeladri Bhuiya, Viktor Schlegel, and Stefan Winkler. 2024. Seemingly Plausi-
ble Distractors in Multi-Hop Reasoning: Are Large Language Models Attentive
Readers?. InProceedings of the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen
(Eds.). Association for Computational Linguistics, Miami, Florida, USA, 2514–2528.
doi:10.18653/v1/2024.emnlp-main.147
[4]Boyi Deng, Wenjie Wang, Fengbin Zhu, Qifan Wang, and Fuli Feng. 2025. CrAM:
Credibility-Aware Attention Modification in LLMs for Combating Misinformation
in RAG.Proceedings of the AAAI Conference on Artificial Intelligence39, 22 (Apr.
2025), 23760–23768. doi:10.1609/aaai.v39i22.34547
[5]Peter Devine. 2025. ALoFTRAG: Automatic Local Fine Tuning for Retrieval
Augmented Generation. arXiv:2501.11929 [cs.LG] https://arxiv.org/abs/2501.
11929
[6]Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia,
Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A
Survey on In-context Learning. InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and
Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida,
USA, 1107–1128. doi:10.18653/v1/2024.emnlp-main.64
[7]Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2024. TRACE the Evidence:
Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented
Generation. InFindings of the Association for Computational Linguistics: EMNLP
2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association
for Computational Linguistics, Miami, Florida, USA, 8472–8494. doi:10.18653/v1/
2024.findings-emnlp.496
[8]Cristina Garbacea, Samuel Carton, Shiyan Yan, and Qiaozhu Mei. 2019. Judge the
Judges: A Large-Scale Evaluation Study of Neural Language Models for Online
Review Generation. InProceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and
Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China,
3968–3981. doi:10.18653/v1/D19-1409
[9]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International
Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625.
doi:10.18653/v1/2020.coling-main.580
[10] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu.
2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy,
Challenges, and Open Questions.ACM Trans. Inf. Syst.43, 2, Article 42 (Jan. 2025),
55 pages. doi:10.1145/3703155
[11] Rohit Kumar Kaliyar, Anurag Goswami, and Pratik Narang. 2021. FakeBERT: Fake
news detection in social media with a BERT-based deep learning approach. 80, 8
(2021). doi:10.1007/s11042-020-10183-2
[12] Rohit Kumar Kaliyar and Navya Singh. 2019. Misinformation Detection on On-
line Social Media-A Survey. In2019 10th International Conference on Comput-
ing, Communication and Networking Technologies (ICCCNT). 1–6. doi:10.1109/
ICCCNT45670.2019.8944587
[13] Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and Davood Rafiei. 2023. Evaluating
Open-Domain Question Answering in the Era of Large Language Models. InPro-
ceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki
(Eds.). Association for Computational Linguistics, Toronto, Canada, 5591–5606.
doi:10.18653/v1/2023.acl-long.307
[14] Ioannis Kazlaris, Efstathios Antoniou, Konstantinos Diamantaras, and Charalam-
pos Bratsas. 2025. From Illusion to Insight: A Taxonomic Survey of Hallucination
Mitigation Techniques in LLMs.AI6, 10 (2025). doi:10.3390/ai6100260
[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.),
Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/paper_
files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[16] Yanyang Li, Shuo Liang, Michael Lyu, and Liwei Wang. 2024. Making Long-
Context Language Models Better Multi-Hop Reasoners. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for
Computational Linguistics, Bangkok, Thailand, 2462–2475. doi:10.18653/v1/2024.
acl-long.135
[17] Chenyu Lin, Yilin Wen, Du Su, Hexiang Tan, Fei Sun, Muhan Chen, Chenfu
Bao, and Zhonghou Lyu. 2026. Resisting Contextual Interference in RAG via
Parametric-Knowledge Reinforcement. arXiv:2506.05154 [cs.CL] https://arxiv.
org/abs/2506.05154
[18] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad,
Wen-tau Yih, and Xilun Chen. 2023. How to Train Your Dragon: Diverse Aug-
mentation Towards Generalizable Dense Retrieval. InFindings of the Associationfor Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Ka-
lika Bali (Eds.). Association for Computational Linguistics, Singapore, 6385–6400.
doi:10.18653/v1/2023.findings-emnlp.423
[19] Xueling Lin and Lei Chen. 2018. Domain-aware multi-truth discovery from
conflicting sources.Proc. VLDB Endow.11, 5 (Oct. 2018), 635–647. doi:10.1145/
3177732.3177739
[20] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models
Use Long Contexts.Transactions of the Association for Computational Linguistics
12 (2024), 157–173. doi:10.1162/tacl_a_00638
[21] Shuyi Liu, Yu-Ming Shang, and Xi Zhang. 2026. TruthfulRAG: Resolving Factual-
level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs.Pro-
ceedings of the AAAI Conference on Artificial Intelligence40, 38 (Mar. 2026), 32168–
32176. doi:10.1609/aaai.v40i38.40489
[22] Tianci Liu, Haoxiang Jiang, Tianze Wang, Ran Xu, Yue Yu, Linjun Zhang, Tuo
Zhao, and Haoyu Wang. 2025. RoseRAG: Robust Retrieval-augmented Generation
with Small-scale LLMs via Margin-aware Preference Optimization. InFindings
of the Association for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce
Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association
for Computational Linguistics, Vienna, Austria, 13036–13054. doi:10.18653/v1/
2025.findings-acl.676
[23] Liangming Pan, Wenhu Chen, Min-Yen Kan, and William Yang Wang. 2023.
Attacking Open-domain Question Answering by Injecting Misinformation. In
Proceedings of the 13th International Joint Conference on Natural Language Pro-
cessing and the 3rd Conference of the Asia-Pacific Chapter of the Association for
Computational Linguistics (Volume 1: Long Papers), Jong C. Park, Yuki Arase,
Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Kris-
nadhi (Eds.). Association for Computational Linguistics, Nusa Dua, Bali, 525–539.
doi:10.18653/v1/2023.ijcnlp-main.35
[24] Ruotong Pan, Boxi Cao, Hongyu Lin, Xianpei Han, Jia Zheng, Sirui Wang, Xunliang
Cai, and Le Sun. 2024. Not All Contexts Are Equal: Teaching LLMs Credibility-
aware Generation. InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung
Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA,
19844–19863. doi:10.18653/v1/2024.emnlp-main.1109
[25] Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and
William Wang. 2023. On the Risk of Misinformation Pollution with Large Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2023,
Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational
Linguistics, Singapore, 1389–1403. doi:10.18653/v1/2023.findings-emnlp.97
[26] Kellin Pelrine, Anne Imouza, Camille Thibault, Meilina Reksoprodjo, Caleb Gupta,
Joel Christoph, Jean-François Godbout, and Reihaneh Rabbany. 2023. Towards
Reliable Misinformation Mitigation: Generalization, Uncertainty, and GPT-4. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for
Computational Linguistics, Singapore, 6399–6429. doi:10.18653/v1/2023.emnlp-
main.395
[27] Dorian Quelle and Alexandre Bovet. 2024. The Perils and Promises of Fact-
Checking with Large Language Models.Frontiers in Artificial IntelligenceVolume
7 - 2024 (2024). doi:10.3389/frai.2024.1341697
[28] Gowtham Ramesh, Makesh Narsimhan Sreedhar, and Junjie Hu. 2023. Single
Sequence Prediction over Reasoning Graphs for Multi-hop QA. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, Toronto, Canada, 11466–11481. doi:10.
18653/v1/2023.acl-long.642
[29] Stephanie Schoch and Yangfeng Ji. 2025. The Good, the Bad, and the Debatable:
A Survey on the Impacts of Data for In-Context Learning. InProceedings of the
2025 Conference on Empirical Methods in Natural Language Processing, Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.).
Association for Computational Linguistics, Suzhou, China, 29798–29812. doi:10.
18653/v1/2025.emnlp-main.1514
[30] Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Pira-
manayagam, Enting Chen, Damien Graux, Andre Melo, Ruofei Lai, Zeren Jiang,
Zhongyang Li, Ye Qi, Yang Ren, Dandan Tu, and Jeff Z. Pan. 2025. GeAR: Graph-
enhanced Agent for Retrieval-augmented Generation. InFindings of the Association
for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational
Linguistics, Vienna, Austria, 12049–12072. doi:10.18653/v1/2025.findings-acl.624
[31] Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, and Claire Cardie.
2025. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG
Under Adversarial Poisoning Attacks. arXiv:2412.16708 [cs.IR] https://arxiv.org/
abs/2412.16708
[32] Haitian Sun, Tania Bedrax-Weiss, and William Cohen. 2019. PullNet: Open Do-
main Question Answering with Iterative Retrieval on Knowledge Bases and Text.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Pro-
cessing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan
(Eds.). Association for Computational Linguistics, Hong Kong, China, 2380–2390.
doi:10.18653/v1/D19-1242
[33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabhar-
wal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
doi:10.1162/tacl_a_00475
[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers,
Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational
10

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
Linguistics, Toronto, Canada, 10014–10037. doi:10.18653/v1/2023.acl-long.557
[35] Vaibhav, Raghuram Mandyam Annasamy, and Eduard Hovy. 2019. Do Sentence
Interactions Matter? Leveraging Sentence Level Representations for Fake News
Classification. InProceedings of the Thirteenth Workshop on Graph-Based Meth-
ods for Natural Language Processing (TextGraphs-13), Dmitry Ustalov, Swapna
Somasundaran, Peter Jansen, Goran Glavaš, Martin Riedl, Mihai Surdeanu, and
Michalis Vazirgiannis (Eds.). Association for Computational Linguistics, Hong
Kong, 134–139. doi:10.18653/v1/D19-5316
[36] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025.
Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge
Conflicts for Large Language Models. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang
Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.).
Association for Computational Linguistics, Vienna, Austria, 30553–30571. doi:10.
18653/v1/2025.acl-long.1476
[37] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2024. Text Embeddings by Weakly-Supervised
Contrastive Pre-training. arXiv:2212.03533 [cs.CL] https://arxiv.org/abs/2212.
03533
[38] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and
Furu Wei. 2024. Improving Text Embeddings with Large Language Models. In
Proceedings of the 62nd Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar
(Eds.). Association for Computational Linguistics, Bangkok, Thailand, 11897–
11916. doi:10.18653/v1/2024.acl-long.642
[39] Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr.
2024. Knowledge Graph Prompting for Multi-Document Question Answering.
Proceedings of the AAAI Conference on Artificial Intelligence38, 17 (Mar. 2024),
19206–19214. doi:10.1609/aaai.v38i17.29889
[40] Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang,
Pengjun Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, Yong Jiang, and Wenjuan
Han. 2024. ChatIE: Zero-Shot Information Extraction via Chatting with ChatGPT.
arXiv:2302.10205 [cs.CL] https://arxiv.org/abs/2302.10205
[41] Wenlong Wu, Haofen Wang, Bohan Li, Peixuan Huang, Xinzhe Zhao, and Lei Liang.
2025. MultiRAG: A Knowledge-Guided Framework for Mitigating Hallucination
in Multi-Source Retrieval Augmented Generation. In2025 IEEE 41st International
Conference on Data Engineering (ICDE). 3070–3083. doi:10.1109/ICDE65448.2025.
00230
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Ellen Riloff,
David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Com-
putational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-1259
[43] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi,
Franziska Roesner, and Yejin Choi. 2019. Defending Against Neural Fake News.
InAdvances in Neural Information Processing Systems, H. Wallach, H. Larochelle,
A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32. Cur-
ran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2019/file/
3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf
[44] Linda Zeng, Rithwik Gupta, Divij Motwani, Yi Zhang, and Diji Yang. 2026. Worse
than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG
Against Misleading Retrievals. arXiv:2502.16101 [cs.AI] https://arxiv.org/abs/
2502.16101
[45] Bowen Zhang and Harold Soh. 2024. Extract, Define, Canonicalize: An LLM-based
Framework for Knowledge Graph Construction. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan,
Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguis-
tics, Miami, Florida, USA, 9820–9836. doi:10.18653/v1/2024.emnlp-main.548
[46] Zhuocheng Zhang, Yang Feng, and Min Zhang. 2025. LevelRAG: Enhancing
Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting
Augmented Searchers. arXiv:2502.18139 [cs.CL] https://arxiv.org/abs/2502.18139
[47] Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li,
Zhicheng Dou, Tsung-Yi Ho, and Philip S. Yu. 2024. Trustworthiness in Retrieval-
Augmented Generation Systems: A Survey. arXiv:2409.10102 [cs.IR] https://arxiv.
org/abs/2409.10102
[48] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. PoisonedRAG:
Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Lan-
guage Models. In34th USENIX Security Symposium, USENIX Security 2025, Seattle,
WA, USA, August 13-15, 2025, Lujo Bauer and Giancarlo Pellegrino (Eds.). USENIX
Association, 3827–3844. https://www.usenix.org/conference/usenixsecurity25/
presentation/zou-poisonedrag
11

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
A Detailed Experimental Setup
In this section, we provide the detailed configurations for compared
methods, the document injection process, and our implementation.
A.1 Compared Methods
ReliableRAG is compared against a diverse set of compared meth-
ods: (1) Naive LLM, which generates answers using only internal
knowledge without retrieval; (2) Vanilla RAG, which follows the
standard RAG pipeline but lacks any mechanism to mitigate misin-
formation; (3) Prompt-based, which incorporates document credi-
bility into prompts to guide the generation of the generator Mgen
without parameter updates; (4) Exclusion, a strategy built upon
prompt-based method to filter out documents below a fixed credibil-
ity threshold; (5) Self-RAG [ 2], fine-tuned on Llama2-7B that utilizes
reflection tokens for self-assessment of retrieval and generation;
(6) CAG [ 24], fine-tuned on Llama2-7B by encoding both content
and credibility scores; (7) Knowledgeable-R1 [ 17], fine-tuned on
Qwen2.5-7B-Instruct to enhance resistance against misleading con-
texts; (8) CrAM [ 4], which dynamically adjusts attention weights
based on document credibility; and (9) TruthfulRAG [ 21], which
employs knowledge graphs and entropy-based filtering to elimi-
nate factual inconsistencies. In addition, we conduct comparative
experiments by employing Llama2-7B and Qwen2.5-7B-Instruct as
the generatorMgen, providing a fair comparison against methods
including Self-RAG, CAG, and Knowledgeable-R1 to further validate
the versatility of our framework (see Appendix F.5).
A.2 Low-credibility Document Injection
To evaluate the effectiveness of ReliableRAG in preventing mis-
information, we construct low-credibility documents that contain
misinformation and support incorrect answers using an LLM (i.e.,
GLM-4-Flash). Following the prior study [4], the LLM is guided by
specific prompts to generate coherent yet misleading content. For
each question, we inject three diverse low-credibility documents
supporting the same erroneous answer to simulate a challenging
misinformation environment. Two evaluation settings are consid-
ered: (1) Ideal setting, where high-credibility (Wikipedia-sourced)
and low-credibility documents are assigned fixed scores of 10 and
1, respectively, to establish an upper-bound performance; and (2)
Evaluator-generated setting, where the evaluator Mevaassigns cred-
ibility scores (ranging from 0 to 10) to each document.
A.3 Implementation Details
Experiments were run on a platform with Python 3.10 and CUDA
12.1. Unless otherwise specified, we employ Llama3-8B-Instruct
as the default LLM and e5-Mistral-7B-Instruct as the default bi-
encoder for computing query–document and query–triple similar-
ities throughout the framework. The platform comprised an Intel
Xeon Gold 6348 CPU, 503 GB of memory, and an NVIDIA A800
80GB GPU. Specifically, generation used a temperature of 1.0 with
deterministic sampling. Whereas compared methods retrieve the
top 5 documents by similarity, ReliableRAG selects the top 5 reason-
ing chains inCfinalbased on their confidence scores. These selected
chains are then utilized to construct the supporting context Z,
which is integrated into the unified input X. This inputXis subse-
quently fed into the generatorM gento produce the final answer.
B Prompting and Context Synthesis Strategies
This section provides supplementary details regarding our frame-
work. First, the detailed content of the assessment framework is
illustrated in Figure 13. Furthermore, we elaborate on the context
synthesis strategies as depicted in Figure 7. Following the construc-
tion of the reasoning chain set Cfinal, we transform the selected
chains into a reliable, evidence-based supporting context Zthat isformatted to be comprehensible to the generator Mgen. To achieve
this, we employ two representative strategies to synthesizeZ:
(1) Triple-based Synthesis (TBS):This strategy directly con-
verts the triples into a concise natural language format. Specifi-
cally, for each triple 𝑡𝑘=(𝑠𝑘,𝑝𝑘,𝑜𝑘)within each reasoning chain
𝑐𝐿
𝑗∈C final, the constituent subject 𝑠𝑘, predicate𝑝𝑘, and object𝑜𝑘are
concatenated into a textual string. These strings are then integrated
to form the supporting context Z. By presenting the generator
Mgenwith this information-dense context (as part of the unified
inputX), this strategy ensures that Mgenis grounded in a concise
information space for final answering.
(2) Source-based Synthesis (SBS):This strategy constructs the
supporting context Zby mapping each triple back to its original
source document. Specifically, for each triple 𝑡𝑘within a reasoning
chain𝑢final
𝑗∈C final, we define a projection function 𝜋(𝑡𝑘)that maps
it back to its corresponding source document 𝑑𝑛∈ D , thereby
restoring the rich semantic details. By treating each such mapping
as a vote for 𝑑𝑛, we employ a frequency-based voting mechanism to
rank the documents based on their total occurrences. The documents
identified through these mappings are then sequentially integrated
according to their ranked order to form Z. This ensures that the
generatorMgenis grounded in the comprehensive content of the
original source documents.
Triple -based Synthesis
Source -based SynthesisThe            
Mona Lisa
Leonardo da 
Vinci
painted by
Da Vinci’s 
notebooks 
a flying 
machine
containContext:
The Mona Lisa painted by Leonardo da Vinci. 
Da Vinci’s notebooks contain a flying machine.
Context: 
Document 1: The Mona Lisa , painted by the 
Renaissance artist Leonardo da Vinci, is a world -
famous masterpiece in the Louvre.
Document 2: Da Vinci’s notebooks contain many 
concepts, including detailed sketches for a flying 
machine.Frequency 
count
Document 
Index1 2 3
Figure 7: Illustration of the Context Synthesis Strategies.
Triple-based Synthesis (TBS) directly converts triples into a
concise natural language format by concatenating their com-
ponents into textual strings to maximize information density.
In contrast, Source-based Synthesis (SBS) utilizes the projec-
tion function 𝜋(·)to map triples back to original source doc-
uments, identifying the most representative evidence docu-
ments through a frequency-based voting mechanism, thereby
restoring the rich semantic details.
C Detailed Case Study
In this section, we present a case study in Table 16 to nstantiate
the internal reasoning process of ReliableRAG when encountering
deceptive misinformation. This case is selected from the HotPotQA
test set. The execution flow and logical evolution of each module
are described as follows:
Step 1: Triple Extraction.This module transforms the source
documents into a set of structured triples. As shown in the Table 16,
the set includes both triples from high-credibility documents (e.g.,
Wikipedia), such as(Leo Harris, notable deal, handshake deal with
12

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
Walt Disney to use Donald Duck as the basis for the Oregon Duck
mascot), and deceptive misinformation from low-credibility doc-
uments, such as(Mickey Mouse’s Legacy in Animation Education,
popular misconception, that Donald Duck was featured in the deal,
not Mickey Mouse). By providing such fine-grained information for
subsequent multi-hop reasoning, this approach effectively mitigates
the lost-in-the-middle issue.
Step 2: Triple Evaluation.Building upon the structured triples
derived from the Triple Extraction module and their respective cred-
ibility scores assigned by the evaluator Meva, this module performs
fine-grained reliability quantification by synthesizing query-triple
semantic relevance and triple credibility. This mechanism ensures
the retention of only the top- 𝐾most reliable and non-redundant
triples, thereby providing a highly trustworthy search space for sub-
sequent chain construction. As illustrated in Table 16, this module
effectively filters out triples containing deceptive misinformation
while preserving the most reliable evidence triples.
Step 3: Chain Construction.Operating on the trustworthy
search space established by the Triple Evaluation module, this mod-
ule employs a beam search strategy to autoregressively construct
robust reasoning chains at each hop. Since the search space has
been filtered for reliability, the construction process naturally avoids
deceptive misinformation. As demonstrated in Table 16, the highest-
ranked candidate chain at the second hop, which consists of(Leo
Harris, notable deal, handshake deal with Walt Disney to use Donald
Duck as the basis for the Oregon Duck mascot)and(Donald Duck, cre-
ation year, 1934), contains sufficient information to directly answer
the question𝑄.
Finally, the final reasoning chains Cfinalare synthesized into the
supporting context Zthrough context synthesis strategies, then
integrated into the unified input Xfor the generatorMgen. By sup-
plying this refined and reliable context Z, ReliableRAG effectively
alleviates the burden on Mgenof integrating raw, potentially decep-
tive misinformation, thereby enabling the generation of the accurate
final answer: “Donald Duck.”
D Definition of Indirect Effect
In this section, following the prior study [ 4] that quantifies the
contribution of specific components by measuring the change in
generation probability, we adopt the Indirect Effect (IE) to validate
the efficacy of the dual-factor perception mechanism and its role in
guiding the construction of reasoning chains. Specifically, we first
establish a baseline probability 𝑃0, which represents the likelihood
of deriving the correct answer for each question 𝑄when the bal-
ancing coefficient 𝛼within the dual-factor perception mechanism
is set to𝛼0=1. To compute this probability, we apply two distinct
strategies to the constructed reasoning chains to synthesize two
corresponding types of supporting contexts Z𝑘(where𝑘∈{ 1,2}).
EachZ𝑘is independently integrated into the input X𝑘and fed to the
generatorMgen, yielding two separate probabilities whose combina-
tion defines 𝑃0. This resulting probability 𝑃0serves as our reference
standard, formulated as follows:
𝑃0={𝑃Mgen(𝑎|X𝑘,𝛼0)}2
𝑘=1(8)
Subsequently, to determine the optimal balance between these fac-
tors, we conduct a grid search for 𝛼on the development sets of
three multi-hop QA datasets, each injected with one low-credibility
document per question. This process identifies the most balanced
parameter configuration , denoted as 𝛼∗, which maximizes the per-
formance of both ReliableRAG-TBS and ReliableRAG-SBS across all
tasks. The probability of generating the correct answer under this
configuration is expressed as:
𝑃1={𝑃Mgen(𝑎|X𝑘,𝛼∗)}2
𝑘=1(9)The influence of the balancing coefficient𝛼is quantified as the IE:
IE𝛼∗=𝑃 1−𝑃0 (10)
Intuitively, the IE distribution across all evaluated questions charac-
terizes the mechanism’s overall efficacy, where a positive shift in
this distribution signifies an enhanced capacity to mitigate misin-
formation.
E Robustness Analysis under
Evaluator-generated Settings
In this section, we analyze the robustness of our proposed frame-
work under evaluator-generated settings. Specifically, we first in-
vestigate the vulnerability of compared methods, followed by a
demonstration of the superior robustness of ReliableRAG.
E.1 Vulnerability of Compared Methods
Under the evaluator-generated setting, we conduct a comparative
analysis by varying the proportion of low-credibility documents
containing deceptive misinformation. As illustrated in Figure 8,
compared methods such as CrAM and Exclusion exhibit a substan-
tial decline in performance as the misinformation ratio increases.
This pronounced degradation confirms the inherent vulnerability
of these approaches when confronted with dense and deceptive
misinformation.
E.2 Robustness of ReliableRAG
As shown in Figure 8, our proposed method maintains robust per-
formance with minimal degradation, significantly outperforming
the compared methods. Notably, the Exact Match (EM) performance
of ReliableRAG-TBS on complex tasks such as MuSiQue remains
remarkably stable even as the proportion of misinformation scales
up. This observed stability potentially relates to the hypothesis
proposed in Section 4.2.2: the dual-factor perception mechanism
within the Triple Evaluation module effectively captures factually
correct triples within low-credibility documents while filtering out
those containing misinformation. By utilizing these correct triples to
construct reasoning chains, our approach successfully prevents the
propagation of misinformation throughout the multi-hop reasoning
process.
F Additional Analysis
In this section, we provide a comprehensive analysis of various
factors influencing the framework’s performance. Specifically, we
first investigate the sensitivity to the maximum chain length 𝐿on
the MuSiQue dataset, followed by an assessment of the impact of
various bi-encoder models. Then, we conduct an ablation study for
ReliableRAG-SBS and analyze the input computational overhead.
Finally, we provide extra evidence regarding the framework’s gen-
eralizability and robustness across various generators and settings.
F.1 Sensitivity to Maximum Chain Length𝐿on
MuSiQue
As illustrated in Figure 9, the results on the MuSiQue development
set align with our previous findings on HotpotQA and 2WikiMulti-
HopQA. Specifically, we observe that when 𝐿>4, the growth of the
average chain length slows significantly, while the EM performance
remains virtually stagnant or even experiences a slight decline. This
cross-dataset consistency further reinforces our earlier observation
that increasing reasoning depth beyond a certain threshold yields
diminishing returns, as additional reasoning hops tend to introduce
deceptive misinformation that compromises overall accuracy.
13

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
1 2 3
Misinformation Document Count0.000.050.100.150.200.250.300.350.400.45EM
(a) EM on HotPotQA1 2 3
Misinformation Document Count0.150.200.250.300.350.400.450.500.55F1 
(b) F1  on HotPotQA
1 2 3
Misinformation Document Count0.000.050.100.150.200.250.300.35EM
(c) EM on 2WikiMultiHopQA1 2 3
Misinformation Document Count0.100.150.200.250.300.350.400.45F1 
(d) F1  on 2WikiMultiHopQA
1 2 3
Misinformation Document Count0.000.050.100.150.20EM
(e) EM on MuSiQue1 2 3
Misinformation Document Count0.050.100.150.200.250.30F1 
(f) F1  on MuSiQue
Exclusion
CAG-7BCrAM
ResilientRAG-TBS
ResilientRAG-SBS
Figure 8: Performance of ReliableRAG versus compared meth-
ods under the evaluator-generated setting, across three multi-
hop QA datasets with varying proportions of low-credibility
documents per question.
1 2 3 4 5 6
Maximum Chain Length L01234Avg. Chain LengthMuSiQue
354045
EM (%)
Avg. Chain Length EM (%)
Figure 9: Impact of𝐿on the multi-hop QA performance and
average chain length of ReliableRAG-TBS on the dev set of
MuSiQue under the ideal setting where one low-credibility
document is injected per question.Table 3: Performance comparison of ReliableRAG-TBS em-
ploying different bi-encoder models for triple retrieval on
the dev sets of three multi-hop QA datasets under the ideal
setting where one low-credibility document is injected per
question.
ModelsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
DRAGON+ 70.0 75.3 66.0 70.0 44.0 49.9
E573.0 79.471.0 73.5 40.0 46.2
E5-Mistral 70.0 77.473.0 76.5 48.0 50.8
Table 4: Performance comparison of ReliableRAG-SBS em-
ploying different bi-encoder models for triple retrieval on
the dev sets of three multi-hop QA datasets under the ideal
setting where one low-credibility document is injected per
question.
ModelsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
DRAGON+ 75.0 79.7 61.0 68.2 43.0 47.8
E5 76.081.770.0 74.4 39.0 46.2
E5-Mistral77.081.474.0 77.8 44.0 49.7
F.2 Impact of Various Bi-encoder Models
To determine the optimal bi-encoder for triple retrieval, we compare
the performance of ReliableRAG using three representative models:
DRAGON+ [ 18], E5 [ 37], and E5-Mistral [ 38]. The evaluation results
on the dev sets of three multi-hop QA datasets for both TBS and
SBS strategies are presented in Table 3 and Table 4. The empirical
results demonstrate that E5-Mistral delivers strong performance
across various datasets and strategies, particularly showing notable
advantages on complex multi-hop scenarios like 2WikiMultiHopQA
and MuSiQue. These findings suggest that E5-Mistral provides su-
perior semantic alignment that effectively facilitates the subsequent
reasoning chain construction. Given its robust overall performance
and better adaptability to challenging datasets, we select E5-Mistral
as the default bi-encoder for all primary experiments.
F.3 Ablation study for ReliableRAG-SBS
Similar to the experimental setup in Section 4.3, we conduct an abla-
tion study on ReliableRAG-SBS. Results in Table 5 demonstrate that
the complete ReliableRAG-SBS outperforms all of its variants across
all test sets. Consistent with the trends observed in ReliableRAG-TBS,
the removal of Triple Extraction on the HotpotQA test set triggers
the most substantial performance drop of 35.8% in EM, confirming
the triples’ pivotal role in organizing information for multi-hop QA.
Furthermore, excluding Triple Evaluation results in a 31.0% de-
crease in EM. Notably, the performance penalty for removing this
stage is significantly more pronounced in the SBS variant than
in TBS. This is primarily because the SBS strategy constructs the
supporting context Zby mapping each triple back to its original
coarse-grained source document. Without the Triple Evaluation
process, the quality of Zdiminishes sharply, which directly impacts
ReliableRAG’s ability to resist deceptive misinformation. Finally,
ReliableRAG-SBS exhibits a distinct performance threshold similar
to that observed in ReliableRAG-TBS. Specifically, without Chain
Construction, the EM performance across all datasets begins to
decline once the number of triples exceeds a certain threshold (typi-
cally between 15 and 20). This again confirms that blindly increasing
the number of triples does not yield further performance gains, high-
lighting the critical role of Chain Construction in organizing these
triples into coherent reasoning chains. Collectively, these findings
further validate the efficacy of each module within our framework
14

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
Table 5: Ablation results of ReliableRAG-SBS on the test sets
of three multi-hop QA datasets. The Top- 𝐾Triple variants
represent the performance where the supporting context Z
is constructed by applying the SBS synthesis strategy to the
top-𝐾most reliable triples without the Chain Construction
module.
MethodHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
w/o Graph Construction 19.30 28.12 12.50 19.93 3.30 12.82
w/o Triple Evaluation 24.10 33.49 15.20 21.76 8.20 18.46
w/o Chain Construction
Top-10 Triples 50.20 63.81 29.70 44.06 23.90 31.69
Top-15 Triples 51.00 64.36 32.60 47.77 25.40 33.47
Top-20 Triples 49.50 63.81 33.30 50.03 24.40 32.92
Top-25 Triples 48.10 62.44 32.90 49.35 24.80 33.48
ReliableRAG-SBS55.10 68.80 43.70 59.67 30.90 38.55
Table 6: Average input context length (in tokens) of Reli-
ableRAG versus compared methods across various LLMs on
the test sets of three multi-hop QA datasets.
Methods LLM HotPotQA 2WikiMultiHopQA MuSiQue
Vanilla RAGGemma-7B 560 542 568
Llama3-8B 580 564 586
Mistral-7B 632 610 638
TruthfulRAG (AAAI’26)Gemma-7B 151 151 156
Llama3-8B 164 165 168
Mistral-7B 173 172 177
ReliableRAG-TBS (ours)Gemma-7B 143 146 157
Llama3-8B 156 160 169
Mistral-7B 162 165 177
ReliableRAG-SBS (ours)Gemma-7B 347 481 447
Llama3-8B 353 485 448
Mistral-7B 388 539 499
and its overall capacity to bolster the robustness and factual relia-
bility of RAG systems.
F.4 Analysis of Input Computational Overhead
Table 6 presents the average input context length (in tokens) across
the test sets of all evaluated datasets, comparing our proposed meth-
ods with other approaches under different LLMs. The results indicate
that ReliableRAG-TBS maintains the most concise input context.
This conciseness stems from its strategy of concatenating the sub-
ject, predicate, and object of each triple into a single textual string.
By transforming structured triples into a single textual string, TBS
focuses on delivering reliable and succinct information. This ensures
that the generator Mgenreceives only the most reliable evidence,
thereby minimizing input computational overhead while maintain-
ing high precision in reasoning.
In contrast, ReliableRAG-SBS exhibits a larger average input con-
text length. This is primarily because the SBS strategy constructs
the supporting context Zby mapping each triple back to its origi-
nal source document. Although this approach increases the input
overhead, it preserves the rich semantic nuances of the source doc-
uments. By providing a more comprehensive textual background
withinZ, the SBS variant prioritizes maximizing the prevention of
deceptive misinformation propagation throughout the multi-hop
process, effectively utilizing a broader context to identify the evi-
dence required for answering questions and resist misinformation.
F.5 Extra Results
To establish a fairer comparison with fine-tuning methods, such
as CAG, Self-RAG, and Knowledgeable-R1, we conducted supple-
mentary experiments by substituting our default generator withLlama-2-7B and Qwen2.5-7B (Instruct). We evaluated these alter-
native generators across both TBS (Table 7) and SBS (Table 8) vari-
ants under the ideal setting. The experimental results demonstrate
that ReliableRAG achieves highly satisfactory and competitive per-
formance across all datasets when integrated with different mod-
els. Notably, when utilizing Qwen2.5-7B under the SBS strategy,
our framework achieves an impressive F1 score of 71.3% on Hot-
PotQA. These findings highlight the strong generalizability and
model-agnostic nature of our framework, proving that it can effec-
tively maintain robust performance without relying on a specific
generator. Furthermore, we conduct an additional experiment by
offline evaluating the credibility score for each document. We then
apply the dual-factor perception mechanism to coarse-grained doc-
uments and perform evaluations under the optimal configuration
(𝛼∗=0.4). The experimental results are presented in Table 9.
Table 7: Impact of Different Generators on ReliableRAG-TBS
performance on the test sets of three multi-hop QA datasets,
where one low-credibility document is injected per question.
ModelsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama-2-7B 45.4 59.1 40.6 50.3 25.4 34.2
Qwen2.5-7B 54.5 68.7 47.9 57.6 28.4 37.3
Table 8: Impact of Different Generators on ReliableRAG-SBS
performance on the test sets of three multi-hop QA datasets,
where one low-credibility document is injected per question.
ModelsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama-2-7B 44.3 56.8 39.4 49.0 23.8 32.0
Qwen2.5-7B 57.3 71.3 49.8 58.5 29.9 39.2
Table 9: Experimental results of applying the dual-factor per-
ception mechanism to coarse-grained documents under the
ideal setting.
HotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
47.9 62.1 24.3 43.4 16.3 24.4
G Comparative Study of Extractor, Evaluator,
and Selector
In this section, we conduct a comparative study focusing on three
key functional units within our framework: the extractor, the eval-
uator, and the selector. Specifically, we first investigate the impact
of different extractors, followed by an analysis of various selectors,
and finally explore the influence of different evaluators.
G.1 Impact of Different Extractors
To evaluate the robustness of our framework across various extrac-
tors, we select Llama3-8B-Instruct, Qwen-7B-Instruct, and Qwen-
14B-Instruct for comparison. For this analysis, we employ an ablated
configuration where the triple evaluation and chain construction
modules are bypassed. Specifically, the top-20 relevant triples are
directly synthesized into the supporting context Zusing either
the TBS or SBS strategy within a single reasoning hop before be-
ing processed by the generator. As shown in Tables 10 and 11, the
performance variance across the three Extractor LLMs is marginal,
with F1 scores on HotPotQA, for instance, converging around 42%
for the TBS strategy and 26% for the SBS strategy. This consistency
suggests that the ReliableRAG framework is model-agnostic, as the
15

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
overall performance is not sensitive to the specific choice of the
Extractor.
Table 10: Impact of Different Extractors on ReliableRAG-TBS
performance on the test sets of multi-hop three QA datasets,
where one low-credibility document is injected per question.
ModelHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama3-8B 29.5 41.6 23.7 31.6 13.4 22.0
Qwen2.5-7B 31.9 42.8 22.7 31.2 15.0 24.1
Qwen2.5-14B 31.6 42.6 23.2 31.4 14.1 23.0
Table 11: Impact of Different Extractors on ReliableRAG-SBS
performance on the test sets of three multi-hop QA datasets,
where one low-credibility document is injected per question.
ModelHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama3-8B 16.9 26.3 7.5 13.1 4.2 14.3
Qwen2.5-7B 17.3 26.3 7.3 13.0 4.4 14.7
Qwen2.5-14B 17.6 27.0 7.7 13.3 4.0 14.3
G.2 Impact of Different Selectors
To evaluate the robustness of ReliableRAG across various selectors,
we conduct comparative experiments using Llama3-8B-Instruct,
Mistral-7B, and Gemma-7B. The performance results for the two
framework variants are reported in Table 12 and Table 13. The exper-
imental data reveals that Llama3-8B-Instruct achieves the highest
scores across all metrics and datasets. In the TBS variant, Llama3-8B
reaches an EM of 53.4% on HotPotQA, outperforming Mistral-7B
and Gemma-7B by a wide margin. This trend extends to the SBS
variant, where Llama3-8B maintains a leading EM of 55.1% on Hot-
PotQA, which is significantly higher than the other LLMs. This
performance gap suggests that superior instruction-following capa-
bilities and internal knowledge density allow the selector to more
accurately identify misinformation, thereby constructing robust
reasoning chains.
Table 12: Impact of different selectors on ReliableRAG-TBS
performance under the ideal setting across the test sets of
three multi-hop QA datasets, with one low-credibility docu-
ment injected per question.
ModelHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama3-8B 53.4 67.0 45.4 56.2 29.1 36.7
Mistral-7B 28.8 38.9 21.4 28.4 11.1 16.6
Gemma-7B 21.2 28.5 17.4 22.2 4.0 8.6
Table 13: Impact of different selectors on ReliableRAG-SBS
performance under the ideal setting across the test sets of
three multi-hop QA datasets, with one low-credibility docu-
ment injected per question.
ModelHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
Llama3-8B 55.1 68.8 43.7 59.7 30.9 38.6
Mistral-7B 40.7 52.0 30.5 44.5 20.0 27.2
Gemma-7B 21.2 28.5 17.4 22.2 4.0 8.6
G.3 Impact of Different Evaluators
To investigate the sensitivity of ReliableRAG to the choice of the
evaluator, we compare the performance of GLM-4-Flash, Mistral-
Small-24B, and Qwen3.5-35B-A3B under both TBS (Table 14) andSBS (Table 15) variants within the evaluator-generated setting. Simi-
lar to the experimental configuration in Section G.1, with the notable
exception that we retain the triple evaluation module for this analy-
sis. The results demonstrate that while models with larger parameter
scales, such as Qwen3.5-35B-A3B, achieve marginally higher metrics
across the datasets, the overall performance differences among the
three evaluators are relatively small. This consistency indicates the
robustness of our framework, suggesting that its effectiveness is
not heavily dependent on a specific underlying LLM. Taking into
account the practical trade-offs regarding API cost, computational
resources, and evaluation efficiency, we ultimately select GLM-4-
Flash as the default evaluator for our primary experiments.
Table 14: Impact of Different Evaluators on ReliableRAG-TBS
performance on the test sets of three multi-hop QA datasets,
where one low-credibility document is injected per question.
MethodsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
GLM-4-Flash 34.4 45.7 23.7 34.2 15.7 22.1
Mistral-Small-24B 38.4 51.0 27.4 39.3 17.8 25.4
Qwen3.5-35B-A3B 41.3 54.5 30.1 42.8 20.4 27.6
Table 15: Impact of Different Evaluators on ReliableRAG-SBS
performance on the test sets of three multi-hop QA datasets,
where one low-credibility document is injected per question.
MethodsHotPotQA 2WikiMultiHopQA MuSiQue
EM (%) F1 (%) EM (%) F1 (%) EM (%) F1 (%)
GLM-4-Flash 26.6 36.5 16.0 25.2 13.8 22.8
Mistral-Small-24B 28.9 39.6 21.0 31.8 16.2 25.1
Qwen3.5-35B-A3B 33.1 45.1 22.9 34.5 17.4 26.7
0 1 2 3 4 5 6 7 8 9 10
Credibility Scores020040060080010001200 Count
(a) Low-credibility Documents
0 1 2 3 4 5 6 7 8 9 10
Credibility Scores02000400060008000100001200014000 Count
(b) Wikipedia-sourced Documents
Figure 10: Distribution of credibility scores generated by the
evaluator on low-credibility and Wikipedia-sourced docu-
ments on the test sets of three multi-hop QA datasets.
H Analysis of Evaluator Capability
In this section, we examine the evaluator’s discriminative capa-
bility in assessing the credibility of both triples and documents
16

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
across three test sets, where three low-credibility documents con-
taining misinformation are injected per question. Specifically, we
first present a distribution analysis of the credibility scores, followed
by a quantitative evaluation using ROC curves.
0 1 2 3 4 5 6 7 8 9 10
Credibility Scores02000400060008000100001200014000 Count
(a) Triples from Low-credibility Documents
0 1 2 3 4 5 6 7 8 9 10
Credibility Scores0250005000075000100000125000150000175000 Count
(b) Triples from Wikipedia-sourced Documents
Figure 11: Distribution of credibility scores generated by
the evaluator on triples extracted from low-credibility and
Wikipedia-sourced documents on the test sets of three multi-
hop QA datasets.
H.1 Distribution Analysis
Figure 10 and Figure 11 illustrate the distribution of credibility scores
for documents and extracted triples, respectively. For Wikipedia-
sourced samples, scores are predominantly concentrated in the high-
value range, exhibiting a clear aggregation trend. Conversely, the
distribution of low-credibility samples is more dispersed, with a
subset of scores remaining in the high-value intervals. This suggests
that certain segments within low-credibility documents may still
contain correct information, which aligns with the conjectures in
Section 4.2.2. Notably, compared to document-level scores, the fine-
grained nature of triples allows capturing correct information even
within deceptive contexts. This granularity is crucial for identify-
ing reliable information in complex, multi-hop QA tasks under the
misleading influence of deceptive misinformation.
H.2 Quantitative Evaluation via ROC Curves
Following the approach of the prior study to examine the capa-
bility of evaluators in assessing the credibility of coarse-grained
documents [ 4], we quantitatively analyze the discriminative per-
formance of the evaluator across varying thresholds. Specifically,
samples are categorized as low-credibility if their scores fall be-
low a given threshold (ranging from 0 to 10) and high-credibility
otherwise. By calculating the True Positive Rate (TPR) and False
Positive Rate (FPR) across the entire range of thresholds, we derive
the Area Under the Curve (AUC). Accordingly, the ROC curves in
Figure 12 illustrate the performance of GLM-4-Flash as the evalua-
tor across three test sets, achieving AUC scores of 0.86 and 0.67 for
document- and triple-level assessments, respectively. It is important
to emphasize that while GLM-4-Flash serves as an assessment tool
within our framework, its performance represents a lower bound
of the overall system’s potential. Since ReliableRAG is designed
to be model-agnostic, the integration of more advanced LLMs as
0.0 0.2 0.4 0.6 0.8 1.0
FPR0.00.20.40.60.81.0TPR
(a) ROC Curve for DocumentsAUC = 0.86
0.0 0.2 0.4 0.6 0.8 1.0
FPR0.00.20.40.60.81.0TPR
(b) ROC Curve for TriplesAUC = 0.67Figure 12: ROC curves of credibility scores generated by the
evaluator for both documents and triples across the test sets
of three multi-hop QA datasets.
evaluators would theoretically yield even higher discriminative pre-
cision. Nevertheless, the current results sufficiently demonstrate
that our framework maintains high reliability and robustness even
with GLM-4-Flash as the evaluator, effectively mitigating the impact
of deceptive misinformation.
17

MM’26, November 10–14, 2026, Rio de Janeiro, Brazil Jiang et al.
Table 16: A case study of the ReliableRAG pipeline on the test set of HotPotQA under the ideal setting.
Stage Content Detail
Documents Wikipedia-sourced Document 1:He was also known for his handshake deal with Walt Disney that permitted the University of
Oregon to use the likeness of Donald Duck as the basis for its mascot, the Oregon Duck. . . .
Wikipedia-sourced Document 2:Donald Duck is a cartoon character created in 1934 at Walt Disney Productions. . . .
Low-credibility Document:Contrary to popular belief, it was not Donald Duck but rather Mickey Mouse who was featured in
this landmark agreement. . . .
Triple Extraction Triples from Wikipedia-sourced Document 1:
(Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot) . . .
Triples from Wikipedia-sourced Document 2:
(Donald Duck, creation year, 1934) . . .
Triples from Low-credibility Document:
(Mickey Mouse’s Legacy in Animation Education, popular misconception, that Donald Duck was featured in the deal, not Mickey
Mouse) . . .
Question Leo A. Harris was also known for his handshake deal with Walt Disney that permitted the University of Oregon to use the likeness
of what cartoon character created in 1934 at Walt Disney Productions?
Triple Evaluation The First Hop (Ranked Triples):
1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot)
2: (Oswald the Lucky Rabbit, replacement, Mickey Mouse, created by Walt Disney in 1928)
3: (Oswald the Lucky Rabbit, rights, taken by Charles Mintz from Walt Disney in 1928) . . .
The Second Hop (Ranked Triples):
1: (Donald Duck, creation year, 1934)
2: (Donald Duck, creator, Walt Disney Productions)
3: (Donald Duck, comic book publication, most published comic book character in the world outside of the superhero genre) . . .
. . .
Chain Construction The First Hop (Ranked Candidate Chains):
1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot)
2: (Oswald the Lucky Rabbit, creators, Ub Iwerks and Walt Disney)
3: (Oswald the Lucky Rabbit, replacement, Mickey Mouse was created by Walt Disney in 1928) . . .
The Second Hop (Ranked Candidate Chains):
1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, creation year, 1934)
2: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, physical characteristics, anthropomorphic white duck with a yellow-orange bill and legs and feet)
3: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, comic book publication, most published comic book character in the world outside of the superhero genre) . . .
. . .
Final Chains 1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, creation year, 1934)
2: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, creation year, 1934), (Donald Duck, creator, Walt Disney Productions), (Donald Duck, comic book publication, most
published comic book character in the world outside of the superhero genre)
3: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),
(Donald Duck, creation year, 1934), (Daisy Duck, creation year, 1940), (Donald Duck, creator, Walt Disney Productions) . . .
Generated Answer Donald Duck
18

ReliableRAG MM’26, November 10–14, 2026, Rio de Janeiro, Brazil
Your task is to evaluate the authenticity of knowledge triplets based on your internal knowledge, reasoning, and inference. T he 
structure of a knowledge triplet is (Subject, Predicate, Object) ,Represents a single factual statement about the relationship between 
entities. I will provide a knowledge triad that may contain accurate information or fictional errors. You need to assign it a Credibility 
Score from 0 to 10, with higher scores indicating higher authenticity and lower scores indicating lower authenticity. Here are 2 
examples, you should follow the output format below:
<example>
Triple: (Albert Einstein, was the first recipient in 1921 of, the Nobel Prize in Physics)
Analysis:
Subject Accuracy: Albert Einstein is correct. Einstein was a real historical figure and a renowned physicist.
Object Accuracy: the Nobel Prize in Physics is valid. This award exists and has been granted since 1901.
Predicate Errors: Albert Einstein was not the first Nobel laureate in Physics. The inaugural prize was awarded in 1901 to Wilhelm 
Conrad Roentgen for his discovery of X -rays. Einstein did receive the Nobel Prize, but it was awarded retrospectively in 1922 (f or the 
year 1921) and recognized his work on the photoelectric effect, not relativity.
Credibility Score: 0
Triple: (Wilhelm Conrad Roentgen, was the first recipient in 1901 of, the Nobel Prize in Physics)
Analysis:
Subject Accuracy: Wilhelm Conrad Roentgen is correct. Roentgen was a German physicist who discovered X -rays.
Object Accuracy: the Nobel Prize in Physics is factual and well -documented.
Predicate Accuracy: Roentgen was indeed the first laureate in Physics, as confirmed by Nobel Prize archives. The inaugural award was 
correctly granted in 1901. The prize honored his discovery of X -rays, which revolutionized medicine and physics.
Credibility Score: 10
</example>
<format>
Triple: ( {subject}, {predicate}, {object} )
Credibility Score:  {score}
</format>
Figure 13: Content of the assessment framework. For instance, in the triple(Albert Einstein, was the first recipient in 1921 of, the
Nobel Prize in Physics), the analysis recognizes that while the entities are authentic, the specific relational claim is inaccurate.
Integrating this analysis, the evaluatorM evagenerates the final credibility score.
19