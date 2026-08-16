# Self-Knowledge Retrieval Augmented Generation Framework for Patent Matching

**Authors**: Jian Zhang, Songlin Lei, Zhuohao Yang, Bangli Liu, Ziwei Wang, Xufeng Weng, Gehan Amaratunga, Yu Lin, Hongwei Wang

**Published**: 2026-08-11 15:08:42

**PDF URL**: [https://arxiv.org/pdf/2608.11030v1](https://arxiv.org/pdf/2608.11030v1)

## Abstract
Patent retrieval and matching based on large language models (LLMs) play a vital role in intellectual property protection. However, due to the complex structure of patent documents, dense technical terminology, and multi-modal information, traditional methods struggle to accurately identify subtle differences between patents. Existing LLM-based patent matching approaches typically rely on domain-specific pretrained or instruction tuning, which often entail high manual labeling costs and catastrophic forgetting. While retrieval-augmented generation (RAG) methods introduce external knowledge they fail to fully leverage LLM's capability to automatically parse patents and mine deep semantic relationships. To address these limitations, this paper proposes a self-knowledge RAG framework that guides LLMs to autonomously extract key technical entities and construct hierarchical ontological structures from patent matching queries, thereby enabling query expansion and precise retrieval. The method integrates the FAISS retrieval with a generative matching mechanism, leveraging self-knowledge to enhance the model's understanding of patent innovations and significantly improve retrieval and matching accuracy. Experimental results demonstrate the outstanding performance of the proposed method on real-world patent datasets, validating its effectiveness and application potential.

## Full Text


<!-- PDF content starts -->

Self-Knowledge Retrieval Augmented Generation
Framework for Patent Matching
Jian Zhanga, Songlin Leib, Zhuohao Yangb, Bangli Liuc, Ziwei Wangc, Xufeng Wengc,
Gehan Amaratungab, Yu Linb, Hongwei Wangabd∗
aSchool of Computer Science and Technology, Zhejiang University, Hangzhou, China
bZJU-UIUC Institute, Zhejiang University, Haining, China
cShaoxing K3i Technology Co. Ltd
dState Key Laboratory of CAD&CG, Zhejiang University, Hangzhou, China
{jianzhang.22, hongweiwang}@zju.edu.cn
Abstract—Patent retrieval and matching based on large lan-
guage models (LLMs) play a vital role in intellectual property
protection. However, due to the complex structure of patent
documents, dense technical terminology, and multi-modal in-
formation, traditional methods struggle to accurately identify
subtle differences between patents. Existing LLM-based patent
matching approaches typically rely on domain-specific pretrained
or instruction tuning, which often entail high manual labeling
costs and catastrophic forgetting. While retrieval-augmented
generation (RAG) methods introduce external knowledge they
fail to fully leverage LLM’s capability to automatically parse
patents and mine deep semantic relationships. To address these
limitations, this paper proposes a self-knowledge RAG framework
that guides LLMs to autonomously extract key technical entities
and construct hierarchical ontological structures from patent
matching queries, thereby enabling query expansion and precise
retrieval. The method integrates the FAISS retrieval with a
generative matching mechanism, leveraging self-knowledge to
enhance the model’s understanding of patent innovations and
significantly improve retrieval and matching accuracy. Experi-
mental results demonstrate the outstanding performance of the
proposed method on real-world patent datasets, validating its
effectiveness and application potential.
Index Terms—Patent Match, Large Language Model, Self-
Knowledge, Retrieval Augmentation Generation
I. INTRODUCTION
Patent retrieval and matching is a critical issue in the field
of intellectual property management [1, 2]. By identifying
key information in patents and detecting patent documents
with similar innovative concepts, it can effectively prevent
intellectual property infringement [3] . However, patent doc-
uments possess complex structures and diverse data types,
often containing multimodal data such as text and images,
along with extensive descriptions using specialized terminol-
ogy. These characteristics pose significant challenges to patent
retrieval [4, 5]. The intricate vocabulary descriptions make
it increasingly difficult to discern subtle differences between
patents, resulting in insufficient precision in patent matching.
With the emergence of large language models (LLMs),
retrieval and matching tasks have seen rapid advancements,
and patent retrieval has gained substantial attention. However,
when applying these LLM technologies to the patent matching
*Corresponding Authordomain, challenges such as domain-specific vocabulary mis-
match and difficulties in understanding technical terms and
invention details persist [6, 7]. To mitigate these issues, current
applications of LLMs primarily focus on domain-specific
knowledge acquisition and matching. Specifically, recent ap-
proaches include pre-training LLMs on specialized corpora [8]
or constructing instruction-tuning datasets for specific tasks
[9]. Nevertheless, these methods typically require substantial
human effort in corpus or dataset construction and carry
the risk of catastrophic forgetting during training [10]. An
alternative strategy involves retrieval-augmented generation
(RAG) using external knowledge [11, 12, 13], leveraging finer-
grained information to reduce noise in the patent retrieval
process and enhance the accuracy of patent analysis. While
parsing patents into triple-based knowledge bases as con-
text for LLMs can improve patent document comprehension
[7, 14, 15], such approaches often overlook the potential of
LLMs to automatically parse patents and construct inter-patent
relationships.
To address these challenges, we propose a self-knowledge
RAG framework that strategically leverages LLMs to au-
tonomously extract knowledge from patent matching queries.
This knowledge subsequently guides both the retrieval process
and the generative matching phase, transforming the LLM
from a passive reasoning tool into an active analysis and
information mining system. The core of our proposed method
is a structured pipeline. LLM performs an in-depth analysis of
the patent matching request, extracting key technical entities
and constructing a hierarchical ontology (e.g., technology,
function, application). The self-mined entities and ontological
structures are used to form an enriched query, enabling more
accurate retrieval of similar patents via the FAISS retriever
[16]. The original patent, retrieved similar patents, and self-
mined entity-ontology structures are integrated into a concise
yet information-rich instruction set. This allows the LLM to
assimilate contextual patent knowledge, enhancing its compre-
hension of innovative aspects and improving the accuracy of
search and generative matching.
The main contributions of this paper are as follows:
•We designed a patent self-knowledge mining strategy
based on LLMs, leveraging the model’s capability to
arXiv:2608.11030v1  [cs.IR]  11 Aug 2026

Patent Match via V anilla LLM
Input
Instruction
Query
CandidateAn integrated rolling and sieving machine based on gravity
sensing for automated tea leaf  ...Please select the most similar patent number from A, B, C and
D. Which number is?
A: An integrated feeding and cutting  ...
B: A turnover machine for overhauling bearings...
.....Input
LLM
Answer
LLM
InputInstruction
QuerySelf-
KnowledgeEntity
OntologyFood Processing Equipment > T ea Leaf
Processing > Automated Rolling and
Sieving Machine...Gravity Sensing System, Rolling Device,
Rolling T able, Rolling Blades,  First Motor ...
LLM
OntologyEntity
AnswerPatent Match via Self-Knowledge RAGFig. 1: The Proposed Framework of Self-Knowledge RAG Patent Matching
automatically extract entity and ontology information
from patents.
•We introduced a self-knowledge guided RAG frame-
work, which enhances FAISS vector index retrieval and
generative matching through self-mined knowledge and
query expansion, surpassing the performance of tradi-
tional methods such as Chain-of-Thought (CoT) and
conventional RAG.
•The effectiveness of our proposed method was validated
on real-world patent datasets, and case studies further
demonstrate the advantages of our approach.
II. RELATEDWORK
Due to the strong diversity and structural complexity of
patent documents, patent matching differs in focus from tradi-
tional patent retrieval. While patent retrieval concentrates on
finding relevant patents, patent matching primarily addresses
the similarity of technical innovations. Early patent matching
methods mainly relied on keyword information [17, 18] or
learned patent innovation similarity through embedded spaces
[19, 20, 21]. These approaches failed to capture innovation
similarities between patents, resulting in limited matching
accuracy.
Recent patent matching research has primarily focused on
frameworks constructed using LLMs, which leverage LLMs’powerful semantic understanding and emergent capabilities
[50,29,61] to learn professional terminology and concepts
through training. MoZi [8] continuously trained LLMs on
patent corpora and conducted instruction tuning for patent-
related questions to enhance LLMs’ understanding of technical
details in patent documents. PatentGPT [9] utilized external
patent knowledge bases for pre-training to help LLMs cap-
ture relationships between patent entities. PatentGPT-Dense
[22] further optimized human-reviewed matching behaviors
through reinforcement learning from human feedback for
matching alignment, demonstrating excellent performance in
patent matching tasks.
Another strategy for utilizing large models adopts RAG
framework [11, 23, 24] to enhance LLMs’ domain-specific
capabilities. However, some studies have shown that noise
in retrieved documents may cause comprehension biases in
LLMs, leading to knowledge conflicts [25, 26] and perfor-
mance degradation [27]. Some research has attempted to
mitigate the impact of retrieval noise through graph-based
RAG methods by refining retrieved documents to prevent LLM
performance deterioration. Although these methods avoid
domain-specific fine-tuning of LLMs and only use external
knowledge such as knowledge triples to enhance response
generation [3, 7, 14, 15], these RAG-based approaches fail to

fully utilize retrieved data for comprehension enhancement and
don’t effectively alleviate conflicts between LLMs’ parametric
memory and external knowledge.
III. METHOD
In the patent matching task, given a query patentp q, the
objective is to select the patent with high similarity of innova-
tion from a set ofNcandidate patentsP={p A, pB, ..., p N},
where each candidate patentp iis associated with a corre-
sponding labely i. The patent matching task involves con-
structing a query framework that outputs the identifiery iof
the candidate patent most relevant to the target patent in terms
of innovative description.
P(˜y|p q,P) =LLM(p q, pA⊕ ··· ⊕p D) (1)
where operator⊕denotes a concatenation operation, the large
language model LLM takes query patentp qand candidate
patentPas input, and outputs the identifiery iof the candidate
patent most similar top q.
This study proposes a self-knowledge RAG framework for
patent matching based on mining with LLMs. The frame-
work guides the LLM to extract key entities and ontological
information from the target patent, constructing a structured
knowledge system through self-mined information to enhance
the accuracy and reliability of retrieval and matching. This
approach eliminates traditional steps such as keyword extrac-
tion and matching in conventional retrieval tasks, adopting
a three-stage paradigm of ”Preparation-Retrieval-Reasoning.”
The overall framework of the method is illustrated in Fig 1.
A. Self-Knowledge Mining from Query Patent
This section introduces a self-mining approach that relies
solely on LLMs to extract key information from query patent
texts, without dependence on external knowledge bases. The
method leverages large models to construct preliminary entity-
ontology relationships.
Given the abstract of a target patent, the LLM is used
to identify entities and ontologies. Specifically, customized
instructions guide the model to extract all critical technical
entities (e.g., specific technological concepts, component mod-
ules, and core methods) and ontological concepts related to
patent technology from the abstract. Entities represent concrete
technical elements extracted as keywords from the patent
abstract. Mathematical representation of entity extraction is
follows:
P(ve(pq, P)) =LLM(Instruct e, pq, P) (2)
where Instruct erepresents the entity extraction instruct
for LLM. When provided with the instruction, the query
patent, and candidate patents, the model generates entity lists
ve(pq, P)for both the query and candidate patents based on
the given requirements.
The ontological information is primarily derived from the
International Patent Classification (IPC) system, a hierarchical
framework essential for organizing and categorizing patents
based on technical innovation. This standardized classification
system systematically groups patents across technical domains,enabling effective analysis and comparison of patents from
diverse industries and languages. By leveraging this prede-
fined conceptual system, abstract categories, attributes, and
relationships are organized in a tree structure. This clarifies and
systematizes the relationships between concepts, facilitating
efficient construction of patent knowledge frameworks and
semantic understanding of core patented technologies. The
process of ontology construction can be formally expressed
as:
P(vo(pq, P)) =LLM(Instruct o, pq, Ve(pq), P, Ve(P)) (3)
where Instruct orepresents instruct guides the construction of
the ontology hierarchy for LLM. Using the input information,
the model completes the ontology constructionvo(pq, P)for
the query and candidate patents.
B. Knowledge-guided Patent Retrieval
In this section, we leverage the self-mined knowledge
obtained from the previous stage and enhance the retrieval
process using the FAISS framework to achieve efficient simi-
larity search.
The entity information acquired in Sec. III-A are con-
catenated to form an expanded query string, which is then
combined with the original abstract text. For example:
TIR(Ve(pq)) =ve
1(pq)⊕ ··· ⊕ve
n(pq), (4)
p∗
q=pq⊕ T IR (5)
where,T IR(Ve(pq))is expanded query patent entity infor-
mation,p∗
qis embeddings of expanded retrieval query patent.
This strategy significantly enriches the semantic information
of the query, addressing potential issues such as incomplete
expression or semantic sparsity in the original query.
Efficient vector similarity retrieval is implemented through
the FAISS framework. A pre-trained language model (in this
study, the BGE model is employed) encodes the abstract of
each patent in the entire candidate patent library. These vectors
are pre-built into a FAISS index, which is optimized for
K-nearest neighbor (KNN) search in large-scale vector sets,
enabling rapid retrieval of the most similar candidate patents.
The pretrained language model encoding represents:
h(p∗
q) =PLM(p∗
q)
h(pi) =PLM(p i),eachp i∈P.(6)
wherePLM(∗)is pretrained language model for encoding re-
trieval information,h(∗)is hidden representation for retrieval
information.
During the retrieval process, the expanded query string is
transformed into a query vector using the same encoding
model. The FAISS index then utilizes cosine similarity to
identify the top-K candidate patents, generating a preliminary
set of candidate patents. The similarity computation can be
formulated as:
C(p∗
q, pi) =h(p∗
q)·h(p i). (7)
whereC(p∗
q, pi)denotes top-K candidate patents list from
similarity retrieval.

C. Context-Aware Generation for Patent Match
In the final decision stage, we integrate all available in-
formation into an instruction set rich with semantic context,
enabling the LLM to perform deep reasoning and generate the
final matching result.
The constructed contextual instruction set primarily includes
the following components:
System Instruction: Explicitly requires the LLM to assume
the role of a patent examiner, conducting comprehensive
matching based on all provided information with rigorous
matching logic.
a) Query Patent:: The original abstract text of the target
patent to be queried.
b) Self-Knowledge Mined Information:: Key entity in-
formation and ontological hierarchy obtained in Sec. III-A.
c) Self-Knowledge Retrieval Context:: The top-K most
similar patents retrieved through the FAISS framework in Sec.
III-B. This provides the LLM with accurate sources of patent
innovation information through similar contexts, enhancing its
reasoning capability regarding patent innovativeness.
d) Task Instruction:: Requires the LLM to evaluate each
candidate patent based on entity keywords and ontological
hierarchy, considering similarities in technical domain, func-
tionality, application, and component structure, to identify the
patent with the highest similarity.
By inputting this combined instruction Instruct ragset into
the LLM, which encompasses micro-level entity information,
macro-level patent ontologies information, and rich contextual
data, the model achieves more accurate matching results com-
pared to conventional RAG methods or simple vector retrieval
approaches. The instruction construction can be formatted as:
Zrag=S(p∗
q, pi)⊕p q⊕P⊕Vo(pq)⊕Vo(P) (8)
P(A(p q, P)) =LLM(Instruct rag,Zrag) (9)
whereP(A(p q, P))is answer of patent matching.
IV. EXPERIMENT ANDANALYSIS
A. Experiment Setup
This section details the dataset, evaluation metrics, baseline
models, and implementation details employed in our study.
1) Dataset:The dataset used in this paper employs Patent-
Match to evaluate different patent matching capabilities. This
dataset collects 1,000 patent matching instances from real
patent documents, with detailed data statistics shown in Table
I. The dataset contains 500 Chinese and 500 English patent
entries, covering 8 categories defined by the International
Patent Classification (IPC). To acquire more comprehensive
patent innovation information through retrieval-augmented
generation, we utilized a curated collection of 300,000 patents
from [28], using BGE-large as the retriever to obtain the top-k
most relevant patents as input for the RAG method.
2) Evaluation Metrics:Following prior work [28], we
adopt Accuracy (Acc) as the primary metric to assess the
effectiveness of patent matching.TABLE I: Dataset Statistic of PatentMatch, including the
count and proportion of data for each International Patent
Classification (IPC).
IPC Description Count Prop
HUM Human Necessities 304 30.4%
OPER Performing Operations; Transporting 264 26.4%
CHEM Chemistry; Metallurgy 60 6.0%
TEXT Textiles; Paper 26 2.6%
CONS Fixed Constructions 40 4.0%
MECH Mechanical Engineering; Lighting; 100 10.0%
Heating; Weapons
PHYS Physics 160 16.0%
ELEC Electricity 46 4.6%
Total 1,000 100%
3) Baselines:We compare the following categories of
baseline approaches: Base Large Language Models: Qwen2-
Instruct-7B, and GLM-4-Chat-9B, Qwen2.5-Instruct-14B.
Domain-specific LLMs: MoZi [8], PatentGPT [9], and
PatentGPT-Dense [22]. Chain-of-Thought (CoT) Reasoning
[29]: This method uses structured prompts to guide models
through step-by-step analysis of patent documents, enhancing
comprehension of innovation points and improving answer
generation. Retrieval-Augmented Generation (RAG) [11]: This
approach retrieves relevant patents as contextual information
to assist LLMs in identifying patents with similar innova-
tions. To evaluate the individual contributions of CoT and
RAG techniques, we applied each method separately to the
tested base LLMs, enabling clear comparison of their relative
effectiveness.
4) Implementation Details:In this study, we selected differ-
ent LLMs as the backbone models, including Qwen2-Instruct-
7B, GLM-4-Chat-9B, and Qwen2.5-Instruct-14B. During the
entity generation and ontology generation phases, we con-
structed entity-ontology relationship graphs using carefully
designed prompts. For the retrieval phase, we employed bge-
large-v1.5 [30] as the encoder, utilizing language-specific
models for Chinese and English respectively to generate word
embeddings for cosine similarity computation. In RAG phase,
we constructed corresponding RAG prompts and utilized the
same backbone models that generated entities and ontologies
for final output generation.
B. Main Results
Our proposed method was validated on a multilingual patent
matching dataset and compared with multiple baseline models.
As shown in Table 2, our approach demonstrates notable
advantages in Chinese, English, and overall performance.
Among the vanilla LLM baselines, significant performance
variations were observed across different models. PatentGPT-
1.0-Dense-70B substantially outperformed smaller parameter
models such as PatentGPT-1.5B and Mozi-7B, achieving
an overall accuracy of 69.1. Among the three open-source
backbone models, GLM-4-Chat-9B and Qwen2.5-Instruct-14B
delivered the best results, with accuracy scores of 65.7 and
66.9, respectively.

TABLE II: Main Results
Method English Chinese Overall
SFT LLM
MoZi-7B [8] 25.8 29.0 27.4
PatentGPT-1.5B [9] 26.2 - -
PatentGPT-1.0-Dense-70B [22] 66.2 72.0 69.1
Qwen2-Instruct-7B
Vanilla LLM 31.4 47.0 39.2
Chain-of-Thought (CoT) 32.8 49.2 41.0
Retrieval-Augmented Generation (RAG)49.268.258.7
Ours 34.469.852.1
GLM-4-Chat-9B
Vanilla LLM 66.4 65.0 65.7
Chain-of-Thought (CoT) 68.0 65.6 66.8
Retrieval-Augmented Generation (RAG) 75.8 69.4 72.6
Ours83.6 79.0 81.3
Qwen2.5-Instruct-14B
Vanilla LLM 63.8 70.0 66.9
Chain-of-Thought (CoT) 64.0 71.2 67.6
Retrieval-Augmented Generation (RAG) 70.8 64.2 67.5
Ours79.2 82.2 80.7
After incorporating CoT strategy, all models showed per-
formance improvements, with Qwen2-Instruct-7B exhibiting
the largest gain—a 1.7% increase over its baseline. In terms
of single-language performance, GLM-4-Chat-9B also showed
significant improvement on the English dataset, rising from
66.4 to 68.0. These results confirm that step-by-step reasoning
via CoT helps LLMs better grasp innovative aspects of patents
and effectively handle complexities in patent texts.
When the RAG strategy was applied, the three backbone
models achieved further performance gains. Notably, Qwen2-
Instruct-7B and GLM-4-Chat-9B showed marked improve-
ments, with Qwen2-Instruct-7B reaching 68.2 on the Chinese
dataset and an overall accuracy of 58.7. On the English dataset,
GLM-4-Chat-9B achieved 75.8, with an overall score of 72.6.
Our proposed method consistently enhanced performance
across both Qwen and GLM backbone models. The most
pronounced improvement was observed with GLM-4-Chat-
9B, which attained 83.6 on the English dataset and 79.0
on the Chinese dataset, resulting in an overall accuracy of
81.3—significantly surpassing all other backbone models.
These outcomes indicate that our method effectively enhances
LLM performance in patent matching tasks, demonstrating
strong applicability and robustness, particularly in multi-
lingual settings, thereby validating its effectiveness.
C. Case Study
This case study compares the prompts used in Vanilla LLM
and our proposed method, as illustrated in Table III. The com-
parison reveals that the Vanilla LLM relies solely on limited
information from query patent, leading to misinterpretations
by LLM and ultimately incorrect matching results. In contrast,
our proposed method leverages internally mined knowledge to
enhance the model’s comprehension of query patent, guiding
it to correctly reason and identify the most accurately matched
patent.V. CONCLUSION
This paper proposes a novel patent matching framework
that enhances the performance of RAG and LLM-based gen-
erative patent matching by mining intrinsic knowledge such as
entities and ontologies from patents themselves. Our method
effectively alleviates issues such as insufficient semantic un-
derstanding and poor retrieval accuracy in patent matching
tasks, while also providing an interpretable approach to patent
relevance by clarifying the rationale behind retrieval decisions.
Future work will focus on dynamic ontology generation and
integrating multimodal patent data (e.g., images and text) to
further improve the model’s adaptability to real-world patent
retrieval and matching scenarios.
ACKNOWLEDGMENT
We would like to thank the anonymous reviewers for
their valuable comments. This work is supported by Na-
tional Key Research and Development Program of China
(2024YFF0907803).
REFERENCES
[1] B. P. Abraham and S. D. Moitra, “Innovation assessment
through patent analysis,”Technovation, 2001.
[2] R. Krestel, R. Chikkamath, C. Hewel, and J. Risch,
“A survey on deep learning for patent analysis,”World
Patent Information, 2021.
[3] Z. Peng and Y . Yang, “Connecting the Dots: Inferring
Patent Phrase Similarity with Retrieved Phrase Graphs,”
inFindings of NAACL, 2024.
[4] M. Lupu, K. Mayer, N. Kando, and A. J. Trippe,Current
challenges in patent information retrieval. Springer,
2017, vol. 37.
[5] Y .-H. Tseng, C.-J. Lin, and Y .-I. Lin, “Text mining
techniques for patent analysis,”Information Processing
& Management, 2007.
[6] L. Jiang, C. Zhang, P. A. Scherz, and S. Goetz, “Can
Large Language Models Generate High-quality Patent
Claims?” inFindings of NAACL, 2025.
[7] L. Siddharth and J. Luo, “Retrieval augmented generation
using engineering design knowledge,”Knowledge-Based
Systems, 2024.
[8] S. Ni, M. Tan, Y . Bai, F. Niu, M. Yang, B. Zhang, R. Xu,
X. Chen, C. Li, and X. Hu, “MoZIP: A Multilingual
Benchmark to Evaluate Large Language Models in In-
tellectual Property,” inLREC-COLING, 2024.
[9] R. Ren, J. Ma, and J. Luo, “Large language model
for patent concept generation,”Advanced Engineering
Informatics, 2025.
[10] V . V . Ramasesh, A. Lewkowycz, and E. Dyer, “Effect of
scale on catastrophic forgetting in neural networks,” in
ICLR, 2022.
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih,
T. Rockt ¨aschel, S. Riedel, and D. Kiela, “Retrieval-
augmented generation for knowledge-intensive NLP
tasks,” inNeurIPS, 2020.

TABLE III: Case Study
Input Data
Instruction: Please select the most similar patent number from A, B, C and D. Which number is? Only choose from options A/B/C/D, without providing
additional analysis.
Query Patent: Socks and a manufacturing method therefor, ... and having the same appearance as an ordinary sock.
Vanilla LLM
Answer: (False)
B. A foot state monitoring method, comprising: ... and hearth care guide information.
Self-Knowledge RAG
Refer to the following patent abstract to answer the subsequent question:
Patent1:Socks and ... outer layer sock body; the inner layer sock body ... sleeving the toes and a connecting ... sock.
Patent2:A foot odor-resistant sock,... left side of the sock body ... first heat insulating layer ... facilitates the healthcare of a user.
Patent3:A sock structure ... the waterproof layer ... the first end (12) and the second end.
Patent4:Socks, and in particular, ... is provided outside the pocket ... is an elongated hole, ... and promotion value.
Patent5:Disclosed are a shoe... preventing the foam layers ... a strong supporting performance ... the production efficiency.
Entity:
Inner Layer Sock Body, Outer Layer Sock Body, Toe Sleeve Part, Connecting Part, Toe Sleeves, Parallel Arrangement, Five-toed Sock, Breathability
Improvement, Fungus Growth Prevention, Manufacturing Method
Ontology:
Original abstract: Apparel Footwear Socks with Special Features,Option A: Apparel Footwear Orthopedic Socks,Option B: Healthcare
Monitoring Foot Temperature Monitoring,Option C: Electronics Medical Devices Finger Probe,Option D: Medical Devices Vascular Devices
Artificial Blood Vessels
Answer:(True)
A. Disclosed are stockings for..., and gradually improve hallux valgus with long-term use.
[12] D. Cai, Y . Wang, L. Liu, and S. Shi, “Recent advances
in retrieval-augmented text generation,” inSIGIR, 2022.
[13] L. M. Amugongo, P. Mascheroni, S. Brooks, S. Doering,
and J. Seidel, “Retrieval augmented generation for large
language models in healthcare: A systematic review,”
PLOS Digital Health, 2025.
[14] J.-M. Chu, H.-C. Lo, J. Hsiang, and C.-C. Cho, “Patent
Response System Optimised for Faithfulness: Procedural
Knowledge Embodiment with Knowledge Graph and
Retrieval Augmented Generation,” inThe 1st Workshop
on KnowLLM, 2024.
[15] S. S. Sakhinana, V . sri vaikunth, and V . Runkana,
“Towards automated patent workflows: AI-orchestrated
multi-agent framework for intellectual property manage-
ment and analysis,” inNeurIPS 2024 Workshop on Open-
World Agents, 2024.
[16] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy,
P.-E. Mazar ´e, M. Lomeli, L. Hosseini, and H. J ´egou,
“The faiss library,” inarXiv, 2024.
[17] G. Cascini and M. Zini, “Measuring patent similarity by
comparing inventions functional trees,” inCAI, 2008.
[18] K. V . Indukuri, A. A. Ambekar, and A. Sureka, “Sim-
ilarity analysis of patent claims using natural language
processing techniques,” inICCIMA, 2007.
[19] G. S. Ascione and V . Sterzi, “A comparative analysis of
embedding models for patent similarity,” inarXiv, 2024.
[20] D. S. Hain, R. Jurowetzki, T. Buchmann, and P. Wolf, “A
text-embedding-based approach to measuring patent-to-
patent technological similarity,”Technological Forecast-
ing and Social Change, 2022.
[21] J. Risch, N. Alder, C. Hewel, and R. Krestel, “Patent-
match: A dataset for matching patent claims & prior art,”
inarXiv, 2020.
[22] Z. Bai, R. Zhang, and L. e. a. Chen, “PatentGPT: A Large
Language Model for Intellectual Property,” inarXiv,2024.
[23] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-
RAG: Learning to retrieve, generate, and critique through
self-reflection,” inICLR, 2024.
[24] Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-
Yu, Y . Yang, J. Callan, and G. Neubig, “Active Retrieval
Augmented Generation,” inEMNLP, 2023.
[25] J. Xie, K. Zhang, J. Chen, R. Lou, and Y . Su, “Adaptive
chameleon or stubborn sloth: Revealing the behavior of
large language models in knowledge conflicts,” inICLR,
2024.
[26] R. Xu, Z. Qi, Z. Guo, C. Wang, H. Wang, Y . Zhang, and
W. Xu, “Knowledge Conflicts for LLMs: A Survey,” in
EMNLP, 2024.
[27] X. Li, S. Mei, Z. Liu, Y . Yan, S. Wang, S. Yu, Z. Zeng,
H. Chen, G. Yu, Z. Liu, M. Sun, and C. Xiong, “RAG-
DDR: Optimizing retrieval-augmented generation using
differentiable data rewards,” inICLR, 2025.
[28] Q. Xiong, Z. Xu, Z. Liu, M. Wang, Z. Chen, Y . Sun,
Y . Gu, X. Li, and G. Yu, “Enhancing the Patent Matching
Capability of Large Language Models via the Memory
Graph,” inSIGIR, 2025.
[29] J. Wei, X. Wang, D. Schuurmans, M. Bosma, b. ichter,
F. Xia, E. Chi, Q. V . Le, and D. Zhou, “Chain-of-thought
prompting elicits reasoning in large language models,” in
NeurIPS, 2022.
[30] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian,
and J.-Y . Nie, “C-Pack: Packed Resources For General
Chinese Embeddings,” inarXiv, 2024.