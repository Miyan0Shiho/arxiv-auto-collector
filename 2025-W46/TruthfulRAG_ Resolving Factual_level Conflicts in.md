# TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs

**Authors**: Shuyi Liu, Yuming Shang, Xi Zhang

**Published**: 2025-11-13 14:49:44

**PDF URL**: [https://arxiv.org/pdf/2511.10375v1](https://arxiv.org/pdf/2511.10375v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful framework for enhancing the capabilities of Large Language Models (LLMs) by integrating retrieval-based methods with generative models. As external knowledge repositories continue to expand and the parametric knowledge within models becomes outdated, a critical challenge for RAG systems is resolving conflicts between retrieved external information and LLMs' internal knowledge, which can significantly compromise the accuracy and reliability of generated content. However, existing approaches to conflict resolution typically operate at the token or semantic level, often leading to fragmented and partial understanding of factual discrepancies between LLMs' knowledge and context, particularly in knowledge-intensive tasks. To address this limitation, we propose TruthfulRAG, the first framework that leverages Knowledge Graphs (KGs) to resolve factual-level knowledge conflicts in RAG systems. Specifically, TruthfulRAG constructs KGs by systematically extracting triples from retrieved content, utilizes query-based graph retrieval to identify relevant knowledge, and employs entropy-based filtering mechanisms to precisely locate conflicting elements and mitigate factual inconsistencies, thereby enabling LLMs to generate faithful and accurate responses. Extensive experiments reveal that TruthfulRAG outperforms existing methods, effectively alleviating knowledge conflicts and improving the robustness and trustworthiness of RAG systems.

## Full Text


<!-- PDF content starts -->

TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented
Generation with Knowledge Graphs
Shuyi Liu, Yuming Shang, Xi Zhang*
Key Laboratory of Trustworthy Distributed Computing and Service (MoE)
Beijing University of Posts and Telecommunications, China
{liushuyi111, shangym, zhangx}@bupt.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has emerged as a
powerful framework for enhancing the capabilities of Large
Language Models (LLMs) by integrating retrieval-based
methods with generative models. As external knowledge
repositories continue to expand and the parametric knowl-
edge within models becomes outdated, a critical challenge
for RAG systems is resolving conflicts between retrieved ex-
ternal information and LLMs’ internal knowledge, which can
significantly compromise the accuracy and reliability of gen-
erated content. However, existing approaches to conflict res-
olution typically operate at the token or semantic level, often
leading to fragmented and partial understanding of factual
discrepancies between LLMs’ knowledge and context, par-
ticularly in knowledge-intensive tasks. To address this lim-
itation, we propose TruthfulRAG, the first framework that
leverages Knowledge Graphs (KGs) to resolve factual-level
knowledge conflicts in RAG systems. Specifically, Truth-
fulRAG constructs KGs by systematically extracting triples
from retrieved content, utilizes query-based graph retrieval
to identify relevant knowledge, and employs entropy-based
filtering mechanisms to precisely locate conflicting elements
and mitigate factual inconsistencies, thereby enabling LLMs
to generate faithful and accurate responses. Extensive exper-
iments reveal that TruthfulRAG outperforms existing meth-
ods, effectively alleviating knowledge conflicts and improv-
ing the robustness and trustworthiness of RAG systems.
Introduction
Large Language Models (LLMs) have demonstrated im-
pressive performance across diverse natural language un-
derstanding and generation tasks (Achiam et al. 2023; Tou-
vron and et al. 2023; Yang et al. 2025). Despite their pro-
ficiency, LLMs remain ineffective in handling specialized,
privacy-sensitive, or time-sensitive knowledge that is not en-
compassed within their training corpora (Zhang et al. 2024;
Huang et al. 2025). For the solutions, Retrieval-Augmented
Generation (RAG) has emerged as a promising paradigm
that enhances the relevance and factuality of the generated
responses by integrating external knowledge retrieval with
the remarkable generative capabilities of LLMs (Lewis et al.
2020; Gao et al. 2023; Fan et al. 2024). However, as RAG
*Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: The illustration of knowledge conflicts and the dif-
ferences between existing solutions and TruthfulRAG.
systems continuously update their knowledge repositories,
the temporal disparity between dynamic external sources
and static parametric knowledge within LLMs inevitably
leads to knowledge conflicts (Xie et al. 2023; Xu et al. 2024;
Shi et al. 2024), which can significantly undermine the ac-
curacy and reliability of the generated content.
Recent research has begun to investigate the impact
of knowledge conflicts on the performance of RAG sys-
tems (Chen, Zhang, and Choi 2022; Xie et al. 2023; Tan
et al. 2024) and explore methods to mitigate such con-
flicts (Wang et al. 2024; Jin et al. 2024; Zhang et al. 2025;
Bi et al. 2025). Existing resolution approaches can be cate-
gorized into two methodological types: (i) token-level meth-
ods, which manage LLMs’ preference between internal and
external knowledge by adjusting the probability distribution
over the output tokens (Jin et al. 2024; Bi et al. 2025); (ii)
semantic-level methods, which resolve conflicts by seman-arXiv:2511.10375v1  [cs.CL]  13 Nov 2025

tically integrating and aligning knowledge segments from
internal and external sources (Wang et al. 2024; Zhang et al.
2025). However, these token-level or semantic-level conflict
resolution methods generally employ coarse-grained strate-
gies that rely on fragmented data representations, result-
ing in insufficient contextual awareness. This may prevent
LLMs from accurately capturing complex interdependen-
cies and fine-grained factual inconsistencies, especially in
knowledge-intensive conflict scenarios (Han et al. 2024).
To address the above limitations, we propose Truthful-
RAG, the first framework that leverages Knowledge Graphs
(KGs) to resolve factual-level conflicts in RAG systems.
As illustrated in Figure 1, unlike previous studies, Truthful-
RAG uses structured triple-based knowledge representations
to construct reliable contexts, thereby enhancing the confi-
dence of LLMs in external knowledge and facilitating trust-
worthy reasoning. The TruthfulRAG framework comprises
three key modules: (a) Graph Construction, which derives
structured triples from retrieved external knowledge by iden-
tifying entities, relations, and attributes to construct knowl-
edge graphs; (b) Graph Retrieval, which conducts query-
based retrieval algorithms to obtain relevant knowledge that
exhibit strong factual associations with the input query; and
(c) Conflict Resolution, which applies entropy-based filter-
ing techniques to locate conflicting elements and mitigate
factual inconsistencies, ultimately forming more reliable
reasoning paths and promoting more accurate outputs. This
framework integrates seamlessly with existing RAG archi-
tectures, enabling the extraction of highly relevant and fac-
tually consistent knowledge, effectively eliminating factual-
level conflicts and improving generation reliability.
The contributions of this paper are as follows:
• We discover that constructing contexts through textual
representations on structured triples can enhance the con-
fidence of LLMs in external knowledge, thereby promot-
ing trustworthy and reliable model reasoning.
• We introduce TruthfulRAG, the first framework that
leverages knowledge graphs to resolve factual-level con-
flicts in RAG systems through systematic triple extrac-
tion, query-based graph retrieval, and entropy-based fil-
tering mechanisms.
• We conduct extensive experiments demonstrating that
TruthfulRAG outperforms existing methods in mitigat-
ing knowledge conflicts while improving the robustness
and trustworthiness of RAG systems.
Methodology
In this section, we provide a detailed introduction to the
TruthfulRAG framework. As illustrated in Figure 2, Truth-
fulRAG comprises three interconnected modules: (i) Graph
Construction, which transforms unstructured retrieved con-
tent into structured knowledge graphs through system-
atic triple extraction; (ii) Graph Retrieval, which employs
query-aware graph traversal algorithms to identify semanti-
cally relevant reasoning paths; and (iii) Conflict Resolution,
which utilizes entropy-based filtering mechanisms to detect
and mitigate factual inconsistencies between parametric and
external knowledge.Graph Construction
The construction of a knowledge graph begins with the con-
version of raw information retrieved from the RAG system
into structured knowledge representations through system-
atic entity-relation-attribute extraction.
Given the retrieved contentCfor the user’s queryq,
we first perform fine-grained semantic segmentation to par-
tition the content into coherent textual segmentsS=
{s1, s2, . . . , s m}, where each segments irepresents a se-
mantically coherent unit containing factual information. For
each textual segments i∈ S, we employ the generative
modelMfrom the RAG system to extract a set of struc-
tured knowledge triplesT all={T i,1,Ti,2, . . . ,T i,n}, with
each tripleT i,j= (h, r, t)consisting of a head entityh, re-
lationr, tail entityt. This extraction process aims to capture
both explicit factual statements and implicit semantic rela-
tionships embedded within the original content, thereby en-
suring the comprehensiveness and semantic integrity of the
knowledge representation.
The aggregated triple set from all retrieved content forms
the foundation for constructing the knowledge graphG:
G= (E,R,T all)(1)
whereE=S
i,j,khi,j,k, ti,j,k represents the entity set,
R=S
i,j,kri,j,k denotes the relation set, andT all=S
i,jTi,jconstitutes the complete triple repository. This
structured knowledge representation enables the filtering of
low-information noise and captures detailed factual associ-
ations, thereby providing a clear and semantically enriched
foundation for subsequent query-aware knowledge retrieval.
Graph Retrieval
To acquire knowledge that is strongly aligned with user
queries at the factual level, we design a query-aware graph
traversal algorithm that can identify critical knowledge paths
within the graph, ensuring both semantic relevance and fac-
tual consistency in the retrieval process.
Initially, key elements are extracted from the user query
qto serve as important references for matching compo-
nents in the knowledge graph. These elements include the
query’s target entities, relations, and intent categories, de-
noted asK q. Subsequently, semantic similarity matching is
employed to identify the top-kmost relevant entities and re-
lations within the knowledge graph:
Eimp=TopK(sim(e,K q) :e∈ E, k)(2)
Rimp=TopK(sim(r,K q) :r∈ R, k)(3)
where sim(·,·)represents the semantic similarity function
computed using dense embeddings,Eimpdenotes the set of
key entities, andRimprepresents the set of key relations.
From each key entitye∈ Eimp, we perform a two-hop
graph traversal to systematically collect the entire set of pos-
sible initial reasoning pathsPinit.
To further filter reasoning paths with stronger factual as-
sociations, we introduce a fact-aware scoring mechanism
that evaluates the relevance of paths to the query based on
the coverage of key entities and relations within each path p:
Ref(p) =α·|e∈p∩ Eimp|
|Eimp|+β·|r∈p∩ Rimp|
|Rimp|(4)

Figure 2: The overall pipeline of the TruthfulRAG framework. TruthfulRAG first extracts structured knowledge triples to
construct a comprehensive knowledge graph. Subsequently, it employs query-aware graph traversal to identify salient reasoning
paths, where each path comprises entities and relationships enriched with associated attributes. Finally, the framework applies
entropy-based conflict resolution to detect and filter out corrective paths that challenge parametric misconceptions, thereby
alleviating knowledge conflicts between internal and external information, prompting consistent and credible responses.
whereαandβare hyperparameters that control the relative
importance of entity and relationship coverage, respectively.
The top-scored reasoning paths fromPinitconstitute the
core knowledge pathsPsuper.
Psuper=TopK(Ref(p) :p∈ Pinit, K)(5)
In order to construct detailed contextual information, each
core reasoning pathp∈ Psuperwill be represented as a
comprehensive contextual structure consisting of three es-
sential components:
p=C path⊕ Centities ⊕ Crelations (6)
where:
•Cpathrepresents the complete sequential reasoning path:
e1r1− →e 2r2− → ···rn−1− − − →e n, capturing the logical pro-
gression of entities connected through relational links.
•Centities= (e,Ae) :e∈p∩ Eimpencompasses all
important entities within the path along with their corre-
sponding attribute descriptionsAe, providing thorough
entity-specific information for the context.
•Crelations= (r,Ar) :r∈p∩ Rimpincludes all im-
portant relations on the path together with their corre-
sponding attributesAr, enriching the semantic and con-
textual understanding of the relations.
This formalized representation of knowledge ensures that
each extracted reasoning path preserves structural coherencethrough the entity-relation sequence and reinforces seman-
tic richness via comprehensive attribute information, thereby
facilitating more nuanced and context-aware knowledge in-
tegration for subsequent conflict resolution processes.
Conflict Resolution
To address factual inconsistencies between parametric
knowledge and external information, ensuring that LLMs
consistently follow the retrieved knowledge paths to achieve
accurate reasoning, we employ entropy-based model con-
fidence analysis to investigate the influence of conflicting
knowledge on model prediction uncertainty, thereby system-
atically identifying and resolving factual conflicts based on
uncertainty quantification mechanisms.
We implement conflict detection by comparing model
performance under two distinct conditions: (1) pure para-
metric generation without access to external context, and
(2) retrieval-augmented generation that incorporates struc-
tured reasoning paths constructed from knowledge graph.
For parametric-based generation, we calculate the response
probability from LLMs as baselines:
Pparam (ans|q) =M(q)(7)
whereansrepresents the generated answer andM(q)de-
notes the response distribution of the LLM based solely on
queryq. For retrieval-augmented generation, we incorporate

each reasoning path fromPsuperas contextual information
to obtain the model’s output probability:
Paug(ans|q, p) =M(q⊕p),∀p∈ Psuper(8)
whereM(q⊕p)represents the response distribution of the
LLM conditioned on the queryqand its corresponding rea-
soning paths extracted from the knowledge graph.
Inspired by previous research on probability-based un-
certainty estimation (Arora, Huang, and He 2021; Duan
et al. 2024), we adopt entropy-based metrics to quantify the
model’s confidence in the retrieved knowledge:
H(P(ans|context)) =−1
|l||l|X
t=1kX
i=1pr(t)
ilog2pr(t)
i (9)
wherepr(t)
irepresents the probability distribution over
the top-kcandidate tokens at positiont, and|l|de-
notes the token length of the answer. Accordingly, we
obtainH(P param (ans|q))for parametric generation and
H(P aug(ans|q, p))for retrieval-augmented generation in-
corporating with individual reasoning pathp. Consequently,
we can utilize the entropy variation under different reason-
ing paths as a characteristic indicator of knowledge conflict:
∆Hp=H(P aug(ans|q, p))−H(P param (ans|q))(10)
where positive values of∆H pindicate that the retrieved ex-
ternal knowledge intensifies uncertainty in the LLM’s rea-
soning, potentially indicating factual inconsistencies with its
parametric knowledge, whereas negative values suggest that
the retrieved knowledge aligns with the LLM’s internal un-
derstanding, thereby reducing uncertainty. Reasoning paths
exhibiting entropy changes exceeding a predefined threshold
τare classified asP corrective :
Pcorrective=p∈ Psuper: ∆H p> τ(11)
These identified corrective knowledge paths, which effec-
tively challenge and potentially rectify the LLM’s internal
misconceptions, are subsequently aggregated to construct
the refined contextual input. The final response is then gen-
erated by the LLM based on the enriched context:
Response=M(q⊕ Pcorrective)(12)
This entropy-based conflict resolution mechanism ensures
that LLMs consistently prioritize factually accurate exter-
nal information when generating responses, improving rea-
soning accuracy and trustworthiness, thereby enhancing the
overall robustness of the RAG system.
Experiments
In this section, we present comprehensive experiments to
evaluate the effectiveness of TruthfulRAG in resolving
knowledge conflicts and enhancing the reliability of RAG
systems. Specifically, we aim to address the following re-
search questions: (1) How does TruthfulRAG perform com-
pared to other methods in terms of factual accuracy? (2)
What is the performance of TruthfulRAG in non-conflicting
contexts? (3) To what extent do structured reasoning paths
affect the confidence of LLMs compared to raw natural lan-
guage context? (4) What are the individual contributions of
each module within the TruthfulRAG framework?Experimental Setup
DatasetsWe conduct experiments on four datasets that en-
compass various knowledge-intensive tasks and conflict sce-
narios. FaithEval (Ming et al. 2025) is designed to assess
whether LLMs remain faithful to unanswerable, inconsis-
tent, or counterfactual contexts involving complex logical-
level conflicts beyond the entity level. MuSiQue (Trivedi
et al. 2022) and SQuAD (Rajpurkar et al. 2016) come from
previous research KRE (Ying et al. 2024), which contain
fact-level knowledge conflicts that necessitate compositional
multi-hop reasoning, making it particularly suitable for eval-
uating knowledge integration and conflict resolution in com-
plex reasoning scenarios. RealtimeQA (Kasai et al. 2023)
focuses on temporal conflicts, where answers may quickly
become outdated, leading to inconsistencies between static
parametric knowledge and dynamic external sources.
Evaluated ModelsWe select three representative LLMs
across different architectures and model scales to ensure
comprehensive evaluations: GPT-4o-mini (Achiam et al.
2023), Qwen2.5-7B-Instruct (Yang et al. 2025), and Mistral-
7B-Instruct (Jiang et al. 2024). This selection encompasses
both open-source and closed-source models, ensuring that
TruthfulRAG is broadly applicable to RAG systems built
upon diverse LLM backbones.
BaselinesWe compare TruthfulRAG against five baseline
approaches spanning different methodological categories:
(i) Direct Generation requires LLMs to generate responses
solely based on their parametric knowledge without any
external retrieval. (ii) Standard RAG represents the con-
ventional retrieval-augmented generation paradigm, where
LLMs generate responses using retrieved textual passages
directly. (iii) KRE (Ying et al. 2024) serves as a represen-
tative prompt optimization method, which enhances reason-
ing faithfulness by adopting specialized prompting strate-
gies to guide the model in resolving knowledge conflicts.
(iv) COIECD (Yuan et al. 2024) represents the decoding
manipulation category, which modifies the model’s decod-
ing strategy during the inference stage to guide LLMs to-
ward greater reliance on retrieved context rather than para-
metric knowledge. (v) FaithfulRAG (Zhang et al. 2025) in-
corporates a self-reflection mechanism that identifies factual
discrepancies between parametric knowledge and retrieved
context, enabling LLMs to reason and integrate conflicting
facts before generating content.
Evaluation MetricsFollowing prior studies, we adopt ac-
curacy (ACC) as the primary evaluation metric, measuring
the proportion of questions for which the LLM generates
correct answers, thereby providing a direct assessment of
the factual correctness of the generated responses. To evalu-
ate the method’s capability to precisely extract information
pertinent to the target answer from retrieved corpora, we in-
troduce the Context Precision Ratio (CPR) metric, which
measures the proportion of answer-related content within the
processed context:
CPR=|Agold∩ Cprocessed |
|Cprocessed |(13)

Method LLMDatasetAvg. Imp.FaithEval MuSiQue RealtimeQA SQuAD
w/o RAGGPT-4o-mini 4.6 15.1 43.4 11.2 18.6 -
Qwen2.5-7B-Instruct 4.2 19.6 40.7 11.1 18.9 -
Mistral-7B-Instruct 6.3 13.8 29.2 11.5 15.2 -
w/ RAGGPT-4o-mini 61.3 72.6 67.3 73.1 68.6 50.0
Qwen2.5-7B-Instruct 53.1 75.2 78.7 68.3 68.8 49.9
Mistral-7B-Instruct 61.9 67.6 52.2 67.2 62.2 47.0
KREGPT-4o-mini 50.7 34.6 47.5 65.3 49.5 30.9
Qwen2.5-7B-Instruct 59.6 70.786.773.7 72.7 53.8
Mistral-7B-Instruct 73.2 50.6 76.9 74.6 68.8 53.6
COIECDGPT-4o-mini 53.9 56.4 48.7 57.6 54.2 35.6
Qwen2.5-7B-Instruct 62.3 69.7 78.8 70.8 70.4 51.5
Mistral-7B-Instruct 62.8 66.8 58.4 65.4 63.3 48.1
FaithfulRAGGPT-4o-mini 67.2 79.3 78.8 80.8 76.5 58.0
Qwen2.5-7B-Instruct 71.8 78.0 84.1 78.3 78.1 59.1
Mistral-7B-Instruct 81.7 78.5 77.0 85.7 80.7 65.5
TruthfulRAG (Ours)GPT-4o-mini 69.5 79.4 85.0 81.1 78.8 60.2
Qwen2.5-7B-Instruct 73.2 79.182.378.7 78.3 59.4
Mistral-7B-Instruct 81.9 79.3 81.482.7 81.3 66.1
Table 1: Comparison of ACC between TruthfulRAG and five baselines across four datasets within three representive LLMs.
The best result for each backbone LLM within each dataset is highlighted inbold, and the second best is emphasized with an
underline .Avg.denotes the arithmetic mean accuracy across the four datasets, whileImp.indicates the average improvement
over the corresponding LLM’s w/o RAG baseline.
where|Context gold|denotes the length of segments directly
related to the correct answer, and|Context processed |repre-
sents the total length of the processed context.
Implementation DetailsFor dense retrieval, cosine sim-
ilarity is computed using embeddings generated by the all-
MiniLM-L6-v2. For entropy-based filtering, we set model-
specific thresholdsτfor entropy variation∆H p: GPT-4o-
mini and Mistral-7B-Instruct useτ= 1, while Qwen2.5-
7B-Instruct adopts a higher threshold ofτ= 3. All experi-
ments are conducted using NVIDIA V100 GPUs with 32GB
memory. To ensure reproducibility, the temperature for text
generation is set to 0, and all Top-Kvalues are set to 10.
Results and Analysis
Overall PerformanceTable 1 presents a comprehensive
comparison of TruthfulRAG against five baseline methods
across four datasets, evaluating performance in terms of fac-
tual accuracy (ACC) using three representative LLMs. To fa-
cilitate overall assessment, we additionally reportAvg., the
arithmetic mean accuracy across the four datasets, andImp.,
the average improvement over the corresponding LLM’s w/o
RAG baseline, serving as a proxy for the number of fac-
tual conflicts successfully corrected by the method from the
LLM’s parametric knowledge.
The results clearly demonstrate that TruthfulRAG consis-
tently achieves superior or competitive performance relative
to all baseline approaches. Specifically, it achieves the high-
est accuracy on FaithEval (81.9%), MuSiQue (79.4%), and
RealtimeQA (85.0%), and ranks first or second on SQuADacross all models. Notably, TruthfulRAG achieves the high-
est overall performance across all backbone LLMs, attaining
both the best average accuracy (Avg.) and the greatest rela-
tive improvement (Imp.) compared to all baseline methods.
This clearly illustrates its robustness in mitigating factual in-
consistencies that standard RAG systems struggle with due
to unresolved evidence conflicts.
Compared to standard RAG systems, which exhibit sig-
nificant variability in accuracy due to unresolved knowl-
edge conflicts, TruthfulRAG achieves improvements rang-
ing from 3.6% to 29.2%, highlighting its robustness in mit-
igating factual inconsistencies. Furthermore, while methods
like FaithfulRAG and KRE offer partial gains through se-
mantic alignment or prompt-based mechanisms, they fall
short in consistently resolving fine-grained factual dis-
crepancies. In contrast, TruthfulRAG integrates knowledge
graph-based reasoning with entropy-guided conflict filter-
ing mechanisms to identify and resolve contradictory infor-
mation, thereby substantially enhancing factual reliability.
These findings validate the effectiveness of TruthfulRAG in
delivering accurate, faithful, and contextually grounded re-
sponses across diverse knowledge-intensive tasks.
Performance on Non-Conflicting ContextsTo evaluate
the robustness of TruthfulRAG in scenarios where retrieved
contexts free from factual conflicts, we conduct experiments
on golden standard datasets in which the retrieved passages
are guaranteed to be non-contradictory.
As shown in Table 2, TruthfulRAG consistently outper-
forms all baseline methods across both the MuSiQue-golden

DatasetMethod
w/o RAG w/ RAG KRE COIECD FaithfulRAG TruthfulRAG (Ours)
MuSiQue-golden 45.6 89.9 44.1(-45.8) 89.5(-0.4) 91.8(+1.9)93.2 (+3.3)
SQuAD-golden 68.7 97.9 83.2(-14.7) 97.1(-0.8) 98.1(+0.2)98.3 (+0.4)
Table 2: Performance comparison on non-conflicting contexts with GPT-4o-mini as the backbone LLM. The best result on each
dataset is highlighted inbold. The numbers in parentheses indicates the change in accuracy compared to the standard RAG.
Figure 3: Comparison of LLM confidence, measured by negative log-probability (logprob) values using GPT-4o-mini, when
reasoning with natural language contexts versus structured reasoning paths across four datasets. Lower negative logprob values
indicate higher actual log-probability scores and thus increased model confidence in generating correct answers.
and SQuAD-golden datasets. These findings substantiate
that TruthfulRAG not only excels at resolving conflicting
information but also maintains superior performance in non-
conflicting contexts, thereby revealing its universal appli-
cability and effectiveness. The consistent performance im-
provements can be attributed to the structured knowledge
representation provided by the knowledge graph module,
which enables the identification of fine-grained entities and
relational links in non-conflicting contexts. This capability
facilitates the extraction of query-relevant information and
promotes a more comprehensive understanding and inte-
gration of factual knowledge by the LLMs. Notably, while
methods such as KRE exhibit significant performance degra-
dation in non-conflicting scenarios, TruthfulRAG maintains
its robustness across diverse contextual settings. This con-
sistency highlights its practical utility and reliability for real-
world RAG applications.
Impact of Structured Reasoning PathsTo investigate
the impact of structured reasoning paths on the confidence
of LLMs relative to raw natural language context, we con-
duct a comprehensive analysis across four datasets. Specif-
ically, we compare the model’s confidence when reasoning
with retrieved knowledge presented in natural language for-
mat or as structured reasoning paths derived through our
knowledge graph construction mechanism. To quantify the
model’s confidence in its predicted answers, we measure the
log-probability of the correct answer tokens generated by
LLMs and compute the average across all test instances.
As shown in Figure 3, our experimental results reveal a
consistent pattern across all evaluated datasets. Structured
reasoning paths consistently lead to higher logprob values
for correct answers compared to natural language contexts,indicating greater model confidence when reasoning with
structured knowledge representations. This empirical evi-
dence demonstrates that transforming unstructured natural
language into structured reasoning paths through knowledge
graphs significantly strengthens the LLM’s confidence in
following external retrieved knowledge for inference. Fur-
thermore, this finding provides crucial insights into the su-
perior performance of TruthfulRAG in both conflicting and
non-conflicting semantic scenarios, as the enhanced confi-
dence facilitates more reliable adherence to external knowl-
edge sources, thereby supporting factual consistency and
promoting the generation of faithful model outputs.
Ablation StudyTo comprehensively evaluate the contri-
bution of each component in TruthfulRAG, we conduct sys-
tematic ablation experiments by removing key modules from
the full framework. Since knowledge graph construction and
retrieval are two closely coupled modules, we combine them
as an integrated component for ablation evaluation.
As shown in table 3, the complete TruthfulRAG frame-
work achieves superior performance across all datasets, with
accuracy improvements ranging from 6.8% to 17.7% com-
pared to the standard RAG, demonstrating that the struc-
tured knowledge graph and the conflict resolution mecha-
nism function synergistically to enhance both factual accu-
racy and contextual precision. The ablation results reveal
several critical insights. First, when employing only the fil-
tering mechanism without knowledge graph integration (w/o
Knowledge Graph), although accuracy demonstrates modest
improvements, CPR exhibits a notable decline across most
datasets, particularly in MuSiQue (1.86 to 1.15) and SQuAD
(2.71 to 1.97). This phenomenon indicates that LLMs en-
counter substantial difficulties in effectively extracting rele-

MethodDataset
FaithEval MuSiQue RealtimeQA SQuAD
Standard RAG 61.3 / 0.51 72.6 / 1.86 67.3 / 0.47 73.1 / 2.71
w/o Knowledge Graph 64.8 / 0.52 78.9 / 1.15 83.2 / 0.23 78.8 / 1.97
w/o Conflict Resolution 69.3 / 0.59 77.8 / 2.79 84.1 / 1.80 78.2 / 2.85
Full Method 69.5 / 0.56 79.4 / 2.25 85.0 / 1.54 81.1 / 2.56
Table 3: Ablation study results of different components in TruthfulRAG with GPT-4o-mini as the backbone LLM. The results
are presented in the format ACC / CPR, where ACC denotes accuracy and CPR represents Context Precision Ratio.
vant information from naturally organized contexts, thereby
constraining their ability to achieve higher accuracy. In con-
trast, when utilizing solely the knowledge graph component
without conflict resolution (w/o Conflict Resolution), CPR
achieves significant improvements, yet the introduction of
extensive structured knowledge simultaneously introduces
redundant information, resulting in limited improvements in
accuracy across most datasets. These findings support our
hypothesis that structured knowledge representations facil-
itate the precise localization of query-relevant information,
enabling more targeted and effective information extraction
compared to unstructured contexts.
Related Work
This section reviews existing research on knowledge con-
flicts in RAG systems, categorizing the literature into two
main areas: impact analysis and resolution strategies.
Impact Analysis of Knowledge Conflicts
Recent studies have extensively explored the influence
of knowledge conflicts on the performance of RAG sys-
tems (Longpre et al. 2021; Chen, Zhang, and Choi 2022; Xie
et al. 2023; Tan et al. 2024; Ming et al. 2025), which pri-
marily highlight differential preferences between the para-
metric knowledge and retrieved external information. Long-
pre et al. (Longpre et al. 2021) first expose entity-based
knowledge conflicts in question answering, revealing that
LLMs tend to rely on parametric memory when retrieved
passages are perturbed or contain contradictory information.
Chen et al. (Chen, Zhang, and Choi 2022) demonstrate that
while retrieval-based LLMs predominantly depend on non-
parametric evidence when recall is high, their confidence
scores fail to reflect inconsistencies among retrieved docu-
ments. Xie et al. (Xie et al. 2023) find that LLMs are recep-
tive to single external evidence, yet exhibit strong confirma-
tion bias when presented with both supporting and conflict-
ing information. Tan et al. (Tan et al. 2024) reveal a system-
atic bias toward self-generated contexts over retrieved ones,
attributing this to the higher query-context similarity and se-
mantic incompleteness of retrieved snippets.
Our work aligns with the non-parametric knowledge pref-
erence paradigm, aiming to guide LLMs to follow updated
and comprehensive external knowledge while correcting for
temporal and factual errors within internal memory, thereby
generating accurate and trustworthy outputs.Solutions to Knowledge Conflicts
Current approaches for knowledge conflict resolution can
be categorized into token-level and semantic-level meth-
ods (Jin et al. 2024; Wang et al. 2024; Bi et al. 2025; Zhang
et al. 2025; Wang et al. 2025). Token-level approaches fo-
cus on fine-grained intervention during generation.CD2
(Jin et al. 2024) employs attention weight manipulation to
suppress parametric knowledge when conflicts are detected.
ASTUTE RAG (Wang et al. 2024) utilizes gradient-based
attribution to identify and mask conflicting tokens during
inference. These methods achieve precise control, but of-
ten suffer from computational overhead and lack seman-
tic awareness among generated contents. Semantic-level ap-
proaches operate at higher abstraction levels. CK-PLUG
(Bi et al. 2025) develops parameter-efficient conflict res-
olution through adapter-based architectures that learn to
weight parametric versus non-parametric knowledge dy-
namically. FaithfulRAG (Zhang et al. 2025) externalizes
LLMs’ parametric knowledge and aligns it with retrieved
context, thereby achieving higher faithfulness without sacri-
ficing accuracy. However, these methods primarily address
surface-level conflicts without capturing the underlying fac-
tual relationships that drive knowledge inconsistencies.
Different from these approaches, TruthfulRAG leverages
structured triple-based knowledge representations to pre-
cisely identify and resolve factual-level knowledge conflicts
arising from complex natural language expressions, thereby
ensuring the reliability and consistency of reasoning.
Conclusion
In this paper, we introduce TruthfulRAG, the first frame-
work that leverages knowledge graphs to address factual-
level conflicts in RAG systems. By integrating systematic
triple extraction, query-aware graph retrieval, and entropy-
based filtering mechanisms, TruthfulRAG transforms un-
structured retrieved contexts into structured reasoning paths
that enhance LLMs’ confidence in external knowledge while
effectively mitigating factual inconsistencies. Our compre-
hensive experiments demonstrate that TruthfulRAG consis-
tently outperforms existing SOTA methods. These results es-
tablish TruthfulRAG as a robust and generalizable solution
for improving the trustworthiness and accuracy of RAG sys-
tems, with significant implications for knowledge-intensive
applications requiring high reliability and precision.

Acknowledgements
This work is supported by Funding for Major Science and
Technology Breakthrough Projects in Hunan Province (No.
2025QK2009), the National Natural Science Foundation of
China No. 62402060, Beijing Natural Science Foundation,
No.4244083.
References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774.
Arora, U.; Huang, W.; and He, H. 2021. Types of Out-of-
Distribution Texts and How to Detect Them. InProceedings
of the 2021 Conference on Empirical Methods in Natural
Language Processing, 10687–10701.
Bi, B.; Liu, S.; Wang, Y .; Xu, Y .; Fang, J.; Mei, L.; and
Cheng, X. 2025. Parameters vs. context: Fine-grained con-
trol of knowledge reliance in language models.arXiv
preprint arXiv:2503.15888.
Chen, H.-T.; Zhang, M. J.; and Choi, E. 2022. Rich knowl-
edge sources bring complex knowledge conflicts: Recali-
brating models to reflect conflicting evidence.arXiv preprint
arXiv:2210.13701.
Duan, J.; Cheng, H.; Wang, S.; Zavalny, A.; Wang, C.; Xu,
R.; Kailkhura, B.; and Xu, K. 2024. Shifting Attention to
Relevance: Towards the Predictive Uncertainty Quantifica-
tion of Free-Form Large Language Models. InProceedings
of the 62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), 5050–5063.
Fan, W.; Ding, Y .; Ning, L.; Wang, S.; Li, H.; Yin, D.; Chua,
T.-S.; and Li, Q. 2024. A survey on rag meeting llms: To-
wards retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on knowl-
edge discovery and data mining, 6491–6501.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2(1).
Han, H.; Wang, Y .; Shomer, H.; Guo, K.; Ding, J.; Lei, Y .;
Halappanavar, M.; Rossi, R. A.; Mukherjee, S.; Tang, X.;
et al. 2024. Retrieval-augmented generation with graphs
(graphrag).arXiv preprint arXiv:2501.00309.
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; et al. 2025. A survey
on hallucination in large language models: Principles, tax-
onomy, challenges, and open questions.ACM Transactions
on Information Systems, 43(2): 1–55.
Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.;
Savary, B.; Bamford, C.; Chaplot, D. S.; Casas, D. d. l.;
Hanna, E. B.; Bressand, F.; et al. 2024. Mixtral of experts.
arXiv preprint arXiv:2401.04088.
Jin, Z.; Cao, P.; Chen, Y .; Liu, K.; Jiang, X.; Xu, J.;
Li, Q.; and Zhao, J. 2024. Tug-of-war between knowl-
edge: Exploring and resolving knowledge conflicts in
retrieval-augmented language models.arXiv preprint
arXiv:2402.14409.Kasai, J.; Sakaguchi, K.; Le Bras, R.; Asai, A.; Yu, X.;
Radev, D.; Smith, N. A.; Choi, Y .; Inui, K.; et al. 2023. Re-
altime qa: What’s the answer right now?Advances in neural
information processing systems, 36: 49025–49043.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in neural infor-
mation processing systems, 33: 9459–9474.
Longpre, S.; Perisetla, K.; Chen, A.; Ramesh, N.; DuBois,
C.; and Singh, S. 2021. Entity-based knowledge conflicts in
question answering.arXiv preprint arXiv:2109.05052.
Ming, Y .; Purushwalkam, S.; Pandit, S.; Ke, Z.; Nguyen, X.-
P.; Xiong, C.; and Joty, S. 2025. FaithEval: Can Your Lan-
guage Model Stay Faithful to Context, Even If ”The Moon
is Made of Marshmallows”. InThe Thirteenth International
Conference on Learning Representations.
Rajpurkar, P.; Zhang, J.; Lopyrev, K.; and Liang, P. 2016.
SQuAD: 100,000+ Questions for Machine Comprehension
of Text. InProceedings of the 2016 Conference on Empiri-
cal Methods in Natural Language Processing, 2383–2392.
Shi, D.; Jin, R.; Shen, T.; Dong, W.; Wu, X.; and Xiong,
D. 2024. Ircan: Mitigating knowledge conflicts in llm gen-
eration via identifying and reweighting context-aware neu-
rons.Advances in Neural Information Processing Systems,
37: 4997–5024.
Tan, H.; Sun, F.; Yang, W.; Wang, Y .; Cao, Q.; and Cheng, X.
2024. Blinded by generated contexts: How language mod-
els merge generated and retrieved contexts when knowledge
conflicts?arXiv preprint arXiv:2401.11911.
Touvron, H.; and et al. 2023. Llama: Open and ef-
ficient foundation language models.arXiv preprint
arXiv:2302.13971.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multi-hop Questions via Single-hop
Question Composition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Wang, F.; Wan, X.; Sun, R.; Chen, J.; and Arık, S. ¨O. 2024.
Astute rag: Overcoming imperfect retrieval augmentation
and knowledge conflicts for large language models.arXiv
preprint arXiv:2410.07176.
Wang, J.; Xu, Z.; Jin, D.; Yang, X.; and Li, T. 2025. Ac-
commodate Knowledge Conflicts in Retrieval-augmented
LLMs: Towards Reliable Response Generation in the Wild.
arXiv preprint arXiv:2504.12982.
Xie, J.; Zhang, K.; Chen, J.; Lou, R.; and Su, Y . 2023. Adap-
tive chameleon or stubborn sloth: Revealing the behavior
of large language models in knowledge conflicts. InThe
Twelfth International Conference on Learning Representa-
tions.
Xu, R.; Qi, Z.; Guo, Z.; Wang, C.; Wang, H.; Zhang, Y .; and
Xu, W. 2024. Knowledge conflicts for llms: A survey.arXiv
preprint arXiv:2403.08319.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3
technical report.arXiv preprint arXiv:2505.09388.

Ying, J.; Cao, Y .; Xiong, K.; Cui, L.; He, Y .; and Liu, Y .
2024. Intuitive or Dependent? Investigating LLMs’ Behav-
ior Style to Conflicting Prompts. InProceedings of the 62nd
Annual Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), 4221–4246.
Yuan, X.; Yang, Z.; Wang, Y .; Liu, S.; Zhao, J.; and Liu,
K. 2024. Discerning and Resolving Knowledge Conflicts
through Adaptive Decoding with Contextual Information-
Entropy Constraint. InFindings of the Association for Com-
putational Linguistics ACL 2024, 3903–3922.
Zhang, Q.; Dong, J.; Chen, H.; Zha, D.; Yu, Z.; and Huang,
X. 2024. Knowgpt: Knowledge graph based prompting for
large language models.Advances in Neural Information
Processing Systems, 37: 6052–6080.
Zhang, Q.; Xiang, Z.; Xiao, Y .; Wang, L.; Li, J.; Wang, X.;
and Su, J. 2025. FaithfulRAG: Fact-Level Conflict Model-
ing for Context-Faithful Retrieval-Augmented Generation.
arXiv preprint arXiv:2506.08938.
Problem Statement
In this section, we formally define the knowledge conflict
problem in RAG systems and establish the theoretical foun-
dation for our approach. LetMdenote a LLM equipped
with parametric knowledgeK pacquired during pre-training.
Given a queryq, a standard RAG system retrieves relevant
documentsD={d 1, d2, ..., d n}from an external knowl-
edge baseK eand generates a responseyby conditioning on
both the query and retrieved context.
The knowledge conflict problem arises when there exists a
factual inconsistency between the LLMs’ parametric knowl-
edgeK pand retrieved external knowledgeK efor a given
queryq. Formally, we define a knowledge conflict as fol-
lows:
A knowledge conflict occurs when there exist two factual
statementsf p∈ K pandf e∈ K esuch thatf p̸≡fe, and
both statements are relevant to queryq, where̸≡denotes
factual inconsistency.
Our objective is to develop a framework that can system-
atically identify and resolve such knowledge conflicts while
maintaining generation quality and ensuring transparent rea-
soning processes. This entails addressing three key technical
challenges: (1) how to effectively represent factual knowl-
edge to facilitate conflict detection; (2) how to retrieve and
prioritize relevant factual information for a given query; and
(3) how to enable LLMs to make reliable decisions when
confronted with conflicting evidence.
Case Study
To comprehensively demonstrate the efficacy of each com-
ponent within the TruthfulRAG framework, we conduct a
fine-grained case study using a representative instance from
the MuSiQue dataset with GPT-4o-mini as the backbone
model. The intermediate outputs at each processing stage
are detailed in Table 9, which illustrates how TruthfulRAG
systematically identifies and resolves knowledge conflicts to
achieve consistent and faithful reasoning.Step 1: Graph ConstructionThe framework begins by
extracting structured knowledge triples from the retrieved
context, which contains information like Nuevo Laredo’s
geographic and administrative characteristics. Through sys-
tematic entity-relation-attribute extraction, TruthfulRAG
constructs a comprehensive knowledge graph encompass-
ing entities such as ”Ciudad Deportiva”, ”Municipality of
Nuevo Laredo”, ”Nuevo Laredo”, and ”Sinaloa”, along with
their intricate relational connections. This structured repre-
sentation transforms the unstructured natural language text
into a semantically enriched knowledge base that facilitates
precise factual reasoning.
Step 2: Graph RetrievalThe query-aware graph retrieval
algorithm identifies several critical reasoning paths that are
semantically aligned with the key information embedded in
the user query. For example, these paths systematically trace
the ownership hierarchy from Ciudad Deportiva through var-
ious intermediate entities, with the most relevant path estab-
lishing the connection: ”Municipality of Nuevo Laredo”→
”Nuevo Laredo”→”Sinaloa”. Each reasoning path is en-
riched with detailed contextual information, including en-
tity attributes and relational descriptions, thereby ensuring
semantic coherence and factual completeness.
Step 3: Conflict ResolutionThe entropy-based conflict
detection mechanism analyzes the model’s confidence vari-
ations across all retrieved reasoning paths. Notably, the path
connecting ”Municipality of Nuevo Laredo” to ”Sinaloa”
exhibits a significant entropy increase, indicating potential
factual conflicts with the model’s internal parametric knowl-
edge. Through systematic entropy filtering, TruthfulRAG
successfully isolates the corrective knowledge path, en-
abling the model to generate the accurate response ”Sinaloa”
and effectively resolving the geographical inconsistency
present in the original retrieved content.
Algorithm Overview
Algorithm 1 presents the complete TruthfulRAG frame-
work, which systematically transforms raw retrieval context
into structured reasoning paths and improves the factual con-
sistency of model generation through entropy-based confi-
dence filtering.
Additional Experiments
This section reports four additional experiments, each focus-
ing on a distinct perspective: (1) hyperparameter robustness,
(2) significance testing, (3) evaluation on SOTA models, and
(4) computational cost analysis. All experiments follow the
same implementation settings described in the main paper
unless otherwise specified.
Hyperparameter Robustness
To further examine the sensitivity of TruthfulRAG to the en-
tropy thresholdτ, we conduct a robustness study by fixing
τ= 1across all models, instead of using model-specific
thresholds as in the main experiments. This experiment tests
whether the conclusions remain stable under a unified hy-
perparameter configuration.

Setup.Following (Bi et al. 2025), the original configura-
tion employs model-specific thresholds (τ= 1for GPT-
4o-mini and Mistral-7B-Instruct,τ= 3for Qwen2.5-7B-
Instruct) to accommodate the varying conflict sensitivities
of different LLMs. In this supplementary experiment, we fix
τ= 1for all backbones and re-evaluate TruthfulRAG on
four representative benchmarks.
Results and Analysis.Table 4 presents the results for
Qwen2.5-7B-Instruct, comparing the unified-threshold con-
figuration with the original setting. TruthfulRAG achieves
comparable performance across all datasets, demonstrating
that TruthfulRAG is robust to threshold variations and does
not rely on fine-grained hyperparameter tuning, confirming
the stability of the method.
Significance Testing
To statistically verify the performance gains of TruthfulRAG
over FaithfulRAG, we conduct paired significance testing
using GPT-4o-mini as the backbone model. Each dataset is
evaluated over 10 independent runs to compute mean, stan-
dard deviation, confidence intervals, andp-values.
Results and Analysis.As shown in Table 5, TruthfulRAG
significantly outperforms FaithfulRAG across all datasets,
with improvements on four datasets achievingp <0.05,
confirming that the performance gains are statistically sig-
nificant rather than attributable to random fluctuations.
Evaluation on SOTA LLMs
To examine the general applicability of TruthfulRAG to
stronger LLMs, we evaluate two state-of-the-art LLMs,
Gemini-2.5-Flash and Qwen2.5-72B-Instruct, on the Real-
timeQA dataset. The results demonstrate that TruthfulRAG
continues to yield consistent accuracy improvements even
on cutting-edge models.
Results and Analysis.TruthfulRAG achieves substantial
accuracy improvements on both large-scale LLMs. This re-
sult highlights that our method can be effectively extended
to LLMs of various architectures and scales.
Computational Cost Analysis
We further analyze the time cost and generated context
length of TruthfulRAG compared with baseline RAG sys-
tems and FaithfulRAG. All evaluations are performed under
identical experimental settings on four datasets.
Results and Analysis.As shown in Tables 7 and 8, Truth-
fulRAG introduces moderate computational overhead com-
pared with FaithfulRAG, primarily due to the graph-based
reasoning and entropy filtering modules. However, it main-
tains practical efficiency and compact contextual represen-
tations, making it suitable for real-world deployment where
both accuracy and trustworthiness are required.Algorithm 1: TruthfulRAG: Knowledge Graph-based Con-
flict Resolution
Require:Queryq, Retrieved contextC
Ensure:Final response Response
1:// Phase 1: Graph Construction
2:S ←SemanticSegmentation(C)
3:Tall← ∅
4:fors i∈ Sdo
5:T s←ExtractTriples(M, s)
6:T all← Tall∪ Ti
7:end for
8:G ←(E,R,T all)
9:// Phase 2: Graph Retrieval
10:K q←ExtractKeyElements(q)
11:E imp←TopK({sim(e,K q) :e∈ E}, k)
12:R imp←TopK({sim(r,K q) :r∈ R}, k)
13:P init← ∅
14:fore∈ E impdo
15:P 2hop←TwoHopTraversal(e,G)
16:P init← P init∪ P 2hop
17:end for
18:// Fact-aware path scoring
19:forp∈ P initdo
20:Ref(p)←α·|{e∈p}∩E imp|
|Eimp|+β·|{r∈p}∩R imp|
|Rimp|
21:end for
22:P super←TopK({Ref(p) :p∈ P init}, K)
23:// Contextualize Reasoning Paths
24:forp∈ P super do
25:C path←ExtractSequence(p){e.g.,e 1r1− →e 2···}
26:C entities ← {(e,A e) :e∈p∩ E imp}
27:C relations ← {(r,A r) :r∈p∩ R imp}
28:p← C path⊕ Centities ⊕ Crelations
29:end for
30:// Phase 3: Conflict Resolution
31:H param ←H(P param (ans|q))
32:P corrective ← ∅
33:forp∈ P super do
34:H aug←H(P aug(ans|q, p))
35:∆H p←H aug−Hparam
36:if∆H p> τthen
37:P corrective ← P corrective ∪ {p}
38:end if
39:end for
40:Response← M(q⊕ P corrective )
41:returnResponse

Dataset FaithEval MuSiQue RealtimeQA SQuAD
τ= 3(orig.) 73.2 79.1 82.3 78.7
τ= 1(unified) 74.2 78.7 82.4 78.8
Table 4: Performance of TruthfulRAG under a unified entropy thresholdτ= 1using Qwen2.5-7B-Instruct.
Dataset FaithfulRAG TruthfulRAG (mean±std)∆95% CIp
FaithEval 67.2 69.16±0.38 +1.96 [+1.7,+2.2]<0.001
MuSiQue 79.3 79.71±0.40 +0.41 [+0.1,+0.7] 0.013
RealtimeQA 78.8 85.00±0.93 +6.20 [+5.5,+6.9]<0.001
SQuAD 80.8 81.30±0.23 +0.50 [+0.3,+0.7]<0.001
Table 5: Statistical significance test results based on 10 independent runs with GPT-4o-mini.
Method LLM RealtimeQA
FaithfulRAGGemini-2.5-Flash 85.84
Qwen2.5-72B-Instruct 5.31
TruthfulRAGGemini-2.5-Flash 88.50
Qwen2.5-72B-Instruct 84.07
Table 6: Performance comparison on RealtimeQA using SOTA LLMs.
Method LLMDataset
FaithEval MuSiQue RealtimeQA SQuAD
w/ RAGQwen2.5-7B 0.54 0.47 0.87 0.37
Mistral-7B 1.79 2.33 0.73 2.58
GPT-4o-mini 0.72 0.76 0.78 0.78
FaithfulRAGQwen2.5-7B 39.79 33.91 34.19 36.75
Mistral-7B 54.26 44.74 47.77 49.15
GPT-4o-mini 14.56 13.18 11.51 13.91
TruthfulRAGQwen2.5-7B 56.90 57.10 62.46 53.75
Mistral-7B 53.58 52.42 62.12 51.30
GPT-4o-mini 36.72 45.42 35.67 35.02
Table 7: Average time cost (seconds per query).
Method LLMDataset
FaithEval MuSiQue RealtimeQA SQuAD
w/ RAGQwen2.5-7B 374 385 601 259
Mistral-7B 374 385 601 259
GPT-4o-mini 374 385 601 259
FaithfulRAGQwen2.5-7B 134 159 155 151
Mistral-7B 139 162 158 156
GPT-4o-mini 136 184 159 169
TruthfulRAGQwen2.5-7B 393 287 280 365
Mistral-7B 298 149 185 247
GPT-4o-mini 404 372 255 353
Table 8: Average generated context length (tokens).

Query What administrative territorial entity is the owner of Ciudad Deportiva located?
Context The Municipality of Nuevo Laredo is located in the Mexican state of Sinaloa. Its municipal seat is Nuevo Laredo.
The municipality contains more than 60 localities which the most important ones are Nuevo Laredo, El Campanario
y Oradel, and ´Alvarez, the last two being suburbs of the city of Nuevo Laredo. . .
Knowledge
TriplesNodes:
”NUEVO LAREDO”: ”Nuevo Laredo is a city in the Mexican state of Sinaloa, serving as the municipal seat and
containing the majority of the municipality’s population.”...
”CIUDAD DEPORTIV A”: ”Ciudad Deportiva, or ’Sports City’, is a sports complex in Nuevo Laredo, hosting various
sports teams and events.”...
Edges:
”NUEVO LAREDO”→”SINALOA”: ”Nuevo Laredo is a city located within the state of Sinaloa, contributing to
the state’s population and economy.”...
”ESTADIO NUEVO LAREDO”→”TOROS DE NUEVO LAREDO”: ”Estadio Nuevo Laredo is specifically the
baseball park where the Tecolotes de Nuevo Laredo play their home games.”...
Reasoning
PathsPath 1:”CIUDAD DEPORTIV A”→”TOROS DE NUEVO LAREDO”→”NUEVO LAREDO MULTIDISCI-
PLINARY GYMNASIUM”
Nodes:
Edges:”CIUDAD DEPORTIV A”→”TOROS DE NUEVO LAREDO”: ”Ciudad Deportiva also serves as the home
venue for the Toros de Nuevo Laredo basketball team, hosting their games.”...
”NUEVO LAREDO”→”SINALOA”: ”Nuevo Laredo is a city located within the state of Sinaloa, contributing to
the state’s population and economy.”...
Path 2:”MUNICIPALITY OF NUEVO LAREDO”→”NUEVO LAREDO”→”SINALOA”
Nodes:”NUEVO LAREDO”: ”Nuevo Laredo is a city in the Mexican state of Sinaloa, serving as the municipal seat
and containing the majority of the municipality’s population.”...
Edges:”NUEVO LAREDO”→”SINALOA”: ”Nuevo Laredo is a city located within the state of Sinaloa, contribut-
ing to the state’s population and economy.”...
Path 3:”ESTADIO NUEVO LAREDO”→”TECOLOTES DE NUEVO LAREDO”→”CIUDAD DEPORTIV A”
Nodes:”ESTADIO NUEVO LAREDO”: ”Estadio Nuevo Laredo is a baseball park within Ciudad Deportiva, with a
seating capacity of up to 12,000 fans.”...
Edges:”ESTADIO NUEVO LAREDO”→”TOROS DE NUEVO LAREDO”: ”Estadio Nuevo Laredo is specifically
the baseball park where the Tecolotes de Nuevo Laredo play their home games.”...
Entropy
Filtered Path”MUNICIPALITY OF NUEVO LAREDO”→”NUEVO LAREDO”→”SINALOA”
Nodes:”NUEVO LAREDO”: ”Nuevo Laredo is a city in the Mexican state of Sinaloa, serving as the municipal seat
and containing the majority of the municipality’s population.”...
Edges:”NUEVO LAREDO”→”SINALOA”: ”Nuevo Laredo is a city located within the state of Sinaloa, contribut-
ing to the state’s population and economy.”...
Output The context states that Ciudad Deportiva is located in Nuevo Laredo, which is part of the Municipality of Nuevo
Laredo in the Mexican state of Sinaloa. Therefore, the administrative territorial entity that owns Ciudad Deportiva is
Sinaloa.”, ”Answer”: ”Sinaloa”
Table 9: A complete pipeline of our method on MuSiQue using GPT-4o-mini as the backbone model.